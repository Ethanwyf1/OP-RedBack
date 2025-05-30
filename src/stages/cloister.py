import streamlit as st
from utils.cache_utils import save_to_cache, load_from_cache, delete_cache, cache_exists

from instancespace.stages.cloister import CloisterStage, CloisterInput
from instancespace.data.options import CloisterOptions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr

import traceback
import io
import os
import tempfile
import zipfile
import json
from typing import Optional

# ---------- UTILITY FUNCTIONS ---------- #

def create_cloister_zip(cloister_output, cloister_options, sifted_output, pilot_output, preprocessing_output):
    """
    Create a ZIP file with all visualization plots and data for download
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Get appropriate feature labels
        if hasattr(preprocessing_output, 'feat_labels'):
            original_feat_labels = preprocessing_output.feat_labels
            
            # Handle different number of features (due to feature selection)
            if len(original_feat_labels) != sifted_output.x.shape[1]:
                if len(original_feat_labels) >= sifted_output.x.shape[1]:
                    feat_labels = original_feat_labels[:sifted_output.x.shape[1]]
                else:
                    feat_labels = [f"Feature {i+1}" for i in range(sifted_output.x.shape[1])]
            else:
                feat_labels = original_feat_labels
        else:
            feat_labels = [f"Feature {i+1}" for i in range(sifted_output.x.shape[1])]
        
        # Save metadata
        metadata = {
            "total_instances": sifted_output.x.shape[0],
            "total_features": sifted_output.x.shape[1],
            "feature_labels": feat_labels,
            "boundary_points": cloister_output.z_edge.shape[0],
            "correlated_boundary_points": cloister_output.z_ecorr.shape[0],
            "projection_error": pilot_output.error,
            "r2": float(np.mean(pilot_output.r2)),
            "correlation_threshold": cloister_options.c_thres,
            "p_value_threshold": cloister_options.p_val
        }
        
        with open(os.path.join(tmp_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save boundary points to CSV
        edge_df = pd.DataFrame(cloister_output.z_edge, columns=["X", "Y"])
        edge_df.to_csv(os.path.join(tmp_dir, "boundary_points.csv"), index=False)
        
        ecorr_df = pd.DataFrame(cloister_output.z_ecorr, columns=["X", "Y"])
        ecorr_df.to_csv(os.path.join(tmp_dir, "correlated_boundary_points.csv"), index=False)
        
        # Save projection matrix and pilot information
        if hasattr(pilot_output, 'a'):
            # Make sure we don't exceed the number of features in the projection matrix
            a_cols = feat_labels
            if len(a_cols) > pilot_output.a.shape[1]:
                a_cols = a_cols[:pilot_output.a.shape[1]]
            elif len(a_cols) < pilot_output.a.shape[1]:
                a_cols = [f"Feature {i+1}" for i in range(pilot_output.a.shape[1])]
                
            a_df = pd.DataFrame(pilot_output.a, index=["Z1", "Z2"], columns=a_cols)
            a_df.to_csv(os.path.join(tmp_dir, "projection_matrix.csv"))
        
        if hasattr(pilot_output, 'pilot_summary'):
            summary_df = pilot_output.pilot_summary
            summary_df.to_csv(os.path.join(tmp_dir, "pilot_summary.csv"), index=False)
            
        # Save projection data
        z_df = pd.DataFrame(pilot_output.z, columns=["Z1", "Z2"])
        z_df.to_csv(os.path.join(tmp_dir, "instances_projection.csv"), index=False)
        
        # Create an info file with parameters used
        with open(os.path.join(tmp_dir, "analysis_parameters.txt"), 'w') as f:
            f.write(f"Cloister Stage Analysis Parameters\n")
            f.write(f"==================================\n\n")
            f.write(f"Correlation Threshold: {cloister_options.c_thres:.3f}\n")
            f.write(f"P-value Threshold: {cloister_options.p_val:.3f}\n")
            
            f.write(f"\nDataset Statistics\n")
            f.write(f"==================\n\n")
            f.write(f"Total Boundary Points: {cloister_output.z_edge.shape[0]}\n")
            f.write(f"Total Correlated Boundary Points: {cloister_output.z_ecorr.shape[0]}\n")
        
        # Create ZIP file
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=tmp_dir)
                    zipf.write(file_path, arcname=arcname)
    
        buffer.seek(0)
        return buffer.getvalue()

# ---------- VISUALIZATION FUNCTIONS ---------- #

def display_correlation_matrix(sifted_output, preprocessing_output, cloister_options):
    """
    Visualize the correlation matrix used for boundary estimation
    """
    # Extract data
    x = sifted_output.x
    
    # Get appropriate feature labels - always use all available features
    if hasattr(preprocessing_output, 'feat_labels'):
        original_feat_labels = preprocessing_output.feat_labels
        
        # If beta attribute exists in sifted_output (from feature selection), use it
        if hasattr(sifted_output, 'beta') and np.sum(sifted_output.beta) == x.shape[1]:
            # Get indices of selected features
            selected_indices = np.where(sifted_output.beta)[0]
            feat_labels = [original_feat_labels[i] for i in selected_indices]
        elif len(original_feat_labels) != x.shape[1]:
            # If number of labels doesn't match number of features, use the first x.shape[1] labels
            # or create generic labels if there aren't enough
            if len(original_feat_labels) >= x.shape[1]:
                feat_labels = original_feat_labels[:x.shape[1]]
            else:
                # Create generic labels - do this silently without a warning
                feat_labels = [f"Feature {i+1}" for i in range(x.shape[1])]
        else:
            feat_labels = original_feat_labels
    else:
        # Create generic labels if no feature labels are available
        feat_labels = [f"Feature {i+1}" for i in range(x.shape[1])]
    
    # Compute correlation matrix
    nfeats = x.shape[1]
    rho = np.zeros((nfeats, nfeats))
    pval = np.zeros((nfeats, nfeats))
    
    for i in range(nfeats):
        for j in range(nfeats):
            if i != j:
                rho[i, j], pval[i, j] = pearsonr(x[:, i], x[:, j])
            else:
                rho[i, j] = 1  # Self-correlation is 1
                pval[i, j] = 0
    
    # Apply p-value threshold
    insignificant_pvals = pval > cloister_options.p_val
    rho_filtered = rho.copy()
    rho_filtered[insignificant_pvals] = 0
    
    # Create DataFrames for display
    corr_df = pd.DataFrame(rho, columns=feat_labels, index=feat_labels)
    corr_filtered_df = pd.DataFrame(rho_filtered, columns=feat_labels, index=feat_labels)
    
    # Create visualizations
    st.subheader("Feature Correlation Matrix")
    
    # Create a toggle for showing filtered or unfiltered
    show_filtered = st.checkbox("Show only statistically significant correlations", value=True)
    
    # Select which correlation matrix to visualize
    display_df = corr_filtered_df if show_filtered else corr_df
    
    # Adjust figure size based on number of features
    fig_width = max(8, nfeats * 0.8)
    fig_height = max(6, nfeats * 0.6)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    mask = np.triu(np.ones_like(display_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Adjust annotation font size based on number of features
    annot_size = max(6, 12 - nfeats * 0.2) if nfeats > 10 else 10
    
    sns.heatmap(display_df, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax,
                annot_kws={'size': annot_size})
    
    title_suffix = " (Filtered by p-value)" if show_filtered else " (Unfiltered)"
    ax.set_title(f"Feature Correlation Matrix{title_suffix}")
    
    # Rotate labels if there are many features
    if nfeats > 10:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Download button for the correlation matrix
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Correlation Matrix",
        data=buf,
        file_name=f"correlation_matrix{'_filtered' if show_filtered else ''}.png",
        mime="image/png",
        key="download_correlation_matrix"
    )
    
    # Display information about correlations
    positive_corr = np.sum(rho_filtered > cloister_options.c_thres)
    negative_corr = np.sum(rho_filtered < -cloister_options.c_thres)
    total_significant = np.sum(~insignificant_pvals) - nfeats  # Subtract diagonal
    
    st.markdown(f"""
    ### Correlation Summary
    
    - **Total Features:** {nfeats}
    - **Total Feature Pairs:** {nfeats * (nfeats - 1) / 2:.0f}
    - **Statistically Significant Correlations:** {total_significant / 2:.0f} ({total_significant / (nfeats * (nfeats - 1)) * 100:.1f}%)
    - **Strong Positive Correlations (> {cloister_options.c_thres}):** {positive_corr / 2:.0f}
    - **Strong Negative Correlations (< -{cloister_options.c_thres}):** {negative_corr / 2:.0f}
    
    *Note: Counts are divided by 2 since the correlation matrix is symmetric.*
    """)

def display_boundary_visualization(cloister_output, sifted_output, pilot_output):
    """
    Visualize the boundary points in the instance space
    """
    # Check if we have valid output data
    if cloister_output.z_edge.size == 0:
        st.warning("No valid boundary points were generated. Try adjusting the thresholds.")
        return
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the projected instances
    z = pilot_output.z  # Use z directly from pilot output
    ax.scatter(z[:, 0], z[:, 1], c='grey', alpha=0.3, label='Instances')
    
    # Plot the boundary and the hull
    if cloister_output.z_edge.size > 0:
        z_edge = cloister_output.z_edge
        
        try:
            hull = ConvexHull(z_edge)
            for simplex in hull.simplices:
                ax.plot(z_edge[simplex, 0], z_edge[simplex, 1], 'r-')
        except:
            pass  # If convex hull fails, just plot points
            
        ax.scatter(z_edge[:, 0], z_edge[:, 1], c='red', s=50, label='Boundary Points')
    
    # Set labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Instance Space with Estimated Boundary')
    ax.legend()
    
    # Display the plot
    st.pyplot(fig)
    
    # Download button for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Boundary Plot",
        data=buf,
        file_name="boundary_visualization.png",
        mime="image/png",
        key="download_boundary_plot"
    )

def display_correlated_boundary(cloister_output, sifted_output, pilot_output):
    """
    Visualize the correlated boundary points in the instance space
    """
    # Check if we have valid output data
    if cloister_output.z_ecorr.size == 0:
        st.warning("No valid correlated boundary points were generated. Try adjusting the correlation threshold.")
        return
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the projected instances
    z = pilot_output.z  # Use z directly from pilot output
    ax.scatter(z[:, 0], z[:, 1], c='grey', alpha=0.3, label='Instances')
    
    # Plot both boundaries for comparison
    if cloister_output.z_edge.size > 0:
        z_edge = cloister_output.z_edge
        
        try:
            hull_edge = ConvexHull(z_edge)
            for simplex in hull_edge.simplices:
                ax.plot(z_edge[simplex, 0], z_edge[simplex, 1], 'r-', alpha=0.3)
        except:
            pass
    
    # Plot the correlated boundary and its hull
    if cloister_output.z_ecorr.size > 0:
        z_ecorr = cloister_output.z_ecorr
        
        try:
            hull_ecorr = ConvexHull(z_ecorr)
            for simplex in hull_ecorr.simplices:
                ax.plot(z_ecorr[simplex, 0], z_ecorr[simplex, 1], 'g-')
        except:
            pass
            
        ax.scatter(z_ecorr[:, 0], z_ecorr[:, 1], c='green', s=50, label='Correlated Boundary Points')
    
    # Set labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Instance Space with Correlation-Based Boundary')
    ax.legend()
    
    # Display the plot
    st.pyplot(fig)
    
    # Download button for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Correlated Boundary Plot",
        data=buf,
        file_name="correlated_boundary_visualization.png",
        mime="image/png",
        key="download_correlated_boundary_plot"
    )

def display_boundary_comparison(cloister_output, sifted_output, pilot_output):
    """
    Visualize a comparison between regular boundary and correlated boundary
    """
    # Check if we have valid output data for both boundaries
    if cloister_output.z_edge.size == 0 or cloister_output.z_ecorr.size == 0:
        st.warning("Cannot compare boundaries. Ensure both boundaries have valid points.")
        return
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot the projected instances in both subplots
    z = pilot_output.z  # Use z directly from pilot output
    ax1.scatter(z[:, 0], z[:, 1], c='grey', alpha=0.3, label='Instances')
    ax2.scatter(z[:, 0], z[:, 1], c='grey', alpha=0.3, label='Instances')
    
    # Plot the regular boundary on the left
    z_edge = cloister_output.z_edge
    try:
        hull_edge = ConvexHull(z_edge)
        for simplex in hull_edge.simplices:
            ax1.plot(z_edge[simplex, 0], z_edge[simplex, 1], 'r-')
        ax1.scatter(z_edge[:, 0], z_edge[:, 1], c='red', s=30, label='Boundary Points')
    except:
        ax1.scatter(z_edge[:, 0], z_edge[:, 1], c='red', s=30, label='Boundary Points')
    
    # Plot the correlated boundary on the right
    z_ecorr = cloister_output.z_ecorr
    try:
        hull_ecorr = ConvexHull(z_ecorr)
        for simplex in hull_ecorr.simplices:
            ax2.plot(z_ecorr[simplex, 0], z_ecorr[simplex, 1], 'g-')
        ax2.scatter(z_ecorr[:, 0], z_ecorr[:, 1], c='green', s=30, label='Correlated Boundary Points')
    except:
        ax2.scatter(z_ecorr[:, 0], z_ecorr[:, 1], c='green', s=30, label='Correlated Boundary Points')
    
    # Set labels and titles
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_title('Full Boundary Estimate')
    ax1.legend()
    
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.set_title('Correlation-Based Boundary Estimate')
    ax2.legend()
    
    # Ensure same scale for both plots
    x_min = min(z[:, 0].min(), z_edge[:, 0].min(), z_ecorr[:, 0].min()) - 0.5
    x_max = max(z[:, 0].max(), z_edge[:, 0].max(), z_ecorr[:, 0].max()) + 0.5
    y_min = min(z[:, 1].min(), z_edge[:, 1].min(), z_ecorr[:, 1].min()) - 0.5
    y_max = max(z[:, 1].max(), z_edge[:, 1].max(), z_ecorr[:, 1].max()) + 0.5
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Download button for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Boundary Comparison",
        data=buf,
        file_name="boundary_comparison.png",
        mime="image/png",
        key="download_boundary_comparison"
    )

def display_boundary_statistics(cloister_output, pilot_output):
    """
    Display statistics about the boundaries and projection
    """
    st.subheader("Instance Space Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Boundary Statistics")
        col1a, col1b, col1c = st.columns(3)
        
        with col1a:
            st.metric("Total Boundary Points", cloister_output.z_edge.shape[0])
        with col1b:
            st.metric("Correlated Boundary Points", cloister_output.z_ecorr.shape[0])
        with col1c:
            if cloister_output.z_edge.shape[0] > 0:
                reduction_percentage = (1 - cloister_output.z_ecorr.shape[0] / cloister_output.z_edge.shape[0]) * 100
                st.metric("Reduction", f"{reduction_percentage:.1f}%")
    
    with col2:
        st.subheader("Projection Quality")
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.metric("Projection Error", f"{pilot_output.error:.4f}")
        with col2b:
            st.metric("Mean R²", f"{np.mean(pilot_output.r2):.4f}")
    
    st.markdown("""
    ### Boundary Explanation
    
    The **Full Boundary** includes all combinations of minimum and maximum feature values without considering feature correlations.
    
    The **Correlated Boundary** refines this by removing boundary points that violate the correlation structure:
    - For positive correlations: removes points where features have opposite signs
    - For negative correlations: removes points where features have the same sign
    
    A significant reduction in boundary points indicates strong correlation structure in your data.
    """)

# ---------- MAIN APPLICATION FUNCTION ---------- #

def show():
    st.header("🧩 Cloister Stage")
    st.write("""
    The Cloister stage estimates the boundary of the instance space using correlation analysis.
    This stage analyzes feature correlations and constructs a convex hull to define the boundary of the feature space.
    """)
    
    # Check if prerequisite outputs exist
    if not cache_exists("preprocessing_output.pkl"):
        st.error("🚫 Preprocessing output not found. Please run the Preprocessing stage first.")
        if st.button("Go to Preprocessing Stage"):
            st.session_state.current_tab = "preprocessing"
            st.rerun()
        return
    
    if not cache_exists("sifted_output.pkl"):
        st.error("🚫 SIFTED output not found. Please run the SIFTED stage first.")
        if st.button("Go to SIFTED Stage"):
            st.session_state.current_tab = "sifted"
            st.rerun()
        return
    
    if not cache_exists("pilot_output.pkl"):
        st.error("🚫 Pilot output not found. Please run the Pilot stage first.")
        if st.button("Go to Pilot Stage"):
            st.session_state.current_tab = "pilot"
            st.rerun()
        return
    
    # Load prerequisite data
    preprocessing_output = load_from_cache("preprocessing_output.pkl")
    sifted_output = load_from_cache("sifted_output.pkl")
    pilot_output = load_from_cache("pilot_output.pkl")
    st.success("✅ Required data loaded successfully!")
    
    # Configuration form
    with st.form("cloister_config_form"):
        st.subheader("Cloister Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            c_thres = st.slider(
                "Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Threshold for considering correlation significant (absolute value)"
            )
        
        with col2:
            p_val = st.slider(
                "P-value Threshold",
                min_value=0.001,
                max_value=0.1,
                value=0.05,
                step=0.001,
                format="%.3f",
                help="Statistical significance threshold for correlations"
            )
        
        submitted = st.form_submit_button("Run Cloister Stage")
    
    if submitted:
        try:
            # Create options object
            cloister_opts = CloisterOptions(
                c_thres=c_thres,
                p_val=p_val
            )
            
            # Create input object - use SIFTED output for the feature matrix
            cloister_in = CloisterInput(
                x=sifted_output.x,            # Feature matrix from SIFTED
                a=pilot_output.a,             # Projection matrix from PILOT
                cloister_options=cloister_opts
            )
            
            # Run Cloister stage with progress indicator
            with st.spinner("Running Cloister stage... This may take a moment."):
                cloister_output = CloisterStage._run(cloister_in)
            
            # Store cloister_options separately instead of adding it to cloister_output
            if cloister_output is not None:
                # Save to session state and cache
                st.session_state["cloister_output"] = cloister_output
                st.session_state["cloister_options"] = cloister_opts  # Store options separately
                st.session_state["ran_cloister"] = True
                save_to_cache(cloister_output, "cloister_output.pkl")
                save_to_cache(cloister_opts, "cloister_options.pkl")  # Save options too
                
                # Show success message
                st.success("✅ Cloister stage run successfully!")
                st.rerun()  # Rerun to show visualizations
        
        except Exception as e:
            st.error(f"Error running Cloister stage: {str(e)}")
            st.code(traceback.format_exc())
    
    # Display results only if the stage has been run successfully
    if st.session_state.get("ran_cloister", False) and cache_exists("cloister_output.pkl"):
        try:
            # Load both cloister_output and cloister_options
            if "cloister_output" not in st.session_state:
                st.session_state["cloister_output"] = load_from_cache("cloister_output.pkl")
            if "cloister_options" not in st.session_state:
                st.session_state["cloister_options"] = load_from_cache("cloister_options.pkl")
            
            cloister_output = st.session_state["cloister_output"]
            cloister_options = st.session_state["cloister_options"]
            
            # Display parameter choices if available
            parameter_cols = st.columns(2)
            
            with parameter_cols[0]:
                st.subheader("Correlation Parameters")
                st.info(f"""
                - **Correlation Threshold:** {cloister_options.c_thres:.2f}
                - **P-value Threshold:** {cloister_options.p_val:.3f}
                """)
            
            with parameter_cols[1]:
                st.subheader("Projection Parameters")
                st.info(f"""
                - **Projection Method:** {getattr(pilot_output, 'method', 'PILOT')}
                - **Projection Quality:** {pilot_output.error:.4f} (error), {np.mean(pilot_output.r2):.4f} (R²)
                """)
            
            # Display boundary statistics
            display_boundary_statistics(cloister_output, pilot_output)
            
            # Create tabs for different visualizations (removed Feature Weights tab)
            tab1, tab2, tab3, tab4 = st.tabs([
                "Correlation Matrix", 
                "Boundary Visualization",
                "Correlated Boundary",
                "Boundary Comparison"
            ])
            
            with tab1:
                st.subheader("🔄 Feature Correlation Analysis")
                with st.expander("❓ What is this visualization?", expanded=False):
                   st.write(
                       "This heatmap shows the Pearson correlation coefficients between all pairs of features. "
                       "Strong positive correlations (close to 1.0) are shown in warm colors, "
                       "while strong negative correlations (close to -1.0) are shown in cool colors. "
                       "The Cloister stage uses these correlations to determine which boundary points are valid based on the correlation structure."
                   )
                display_correlation_matrix(sifted_output, preprocessing_output, cloister_options)
           
            with tab2:
               st.subheader("🔲 Full Boundary Visualization")
               with st.expander("❓ What is this visualization?", expanded=False):
                   st.write(
                       "This visualization shows the full boundary of the instance space. "
                       "The red points represent the boundary points generated using all combinations of minimum and maximum feature values. "
                       "The red line represents the convex hull of these boundary points, which defines the estimated boundary of the instance space."
                   )
               display_boundary_visualization(cloister_output, sifted_output, pilot_output)
           
            with tab3:
               st.subheader("🔰 Correlation-Based Boundary")
               with st.expander("❓ What is this visualization?", expanded=False):
                   st.write(
                       "This visualization shows the refined boundary based on correlation analysis. "
                       "The green points represent boundary points that respect the correlation structure of the features. "
                       "The green line represents the convex hull of these points, which defines a more accurate boundary that considers feature correlations."
                   )
               display_correlated_boundary(cloister_output, sifted_output, pilot_output)
           
            with tab4:
               st.subheader("🔍 Boundary Comparison")
               with st.expander("❓ What is this visualization?", expanded=False):
                   st.write(
                       "This visualization compares the full boundary (left) with the correlation-based boundary (right). "
                       "The comparison helps visualize how much the boundary changes when considering feature correlations. "
                       "A significant difference indicates that the correlation structure is important for accurately defining the instance space boundary."
                   )
               display_boundary_comparison(cloister_output, sifted_output, pilot_output)
           
           # Add option to download comprehensive results
            st.subheader("Download All Results")
           
            zip_data = create_cloister_zip(cloister_output, cloister_options, sifted_output, pilot_output, preprocessing_output)
           
            st.download_button(
               label="📥 Download Complete Analysis (ZIP)",
               data=zip_data,
               file_name="cloister_analysis.zip",
               mime="application/zip",
               help="Download all visualizations, boundary points, and parameter information",
               key="download_complete_analysis_zip"
           )
           
           # Cache management
            st.subheader("🗑️ Cache Management")
            if st.button("❌ Delete Cloister Cache"):
               success1 = delete_cache("cloister_output.pkl")
               success2 = delete_cache("cloister_options.pkl")
               if success1 or success2:
                   st.success("🗑️ Cloister cache deleted.")
                   if "cloister_output" in st.session_state:
                       del st.session_state["cloister_output"]
                   if "cloister_options" in st.session_state:
                       del st.session_state["cloister_options"]
                   if "ran_cloister" in st.session_state:
                       del st.session_state["ran_cloister"]
                   st.rerun()
               else:
                   st.warning("⚠️ No cache file found to delete.")
       
        except Exception as e:
           st.error(f"Error displaying Cloister results: {str(e)}")
           st.code(traceback.format_exc())
    else:
        if cache_exists("cloister_output.pkl") and not st.session_state.get("ran_cloister", False):
           st.session_state["ran_cloister"] = True
           st.rerun()

if __name__ == "__main__":
   show()
