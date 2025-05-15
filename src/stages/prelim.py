import io
import json
import os
import tempfile
import traceback
import zipfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from instancespace.data.options import PrelimOptions, SelvarsOptions
from instancespace.stages.prelim import PrelimInput, PrelimStage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.cache_utils import (cache_exists, delete_cache, load_from_cache,
                               save_to_cache)

# ---------- UTILITY FUNCTIONS ---------- #


def preprocess_performance_matrix(y):
    """
    Preprocess the performance matrix to ensure type compatibility.
    Handles NaN and infinity values appropriately.
    """
    # Convert to float64
    y_processed = y.astype(np.float64)

    # Replace any NaN or infinity values
    y_processed = np.nan_to_num(
        y_processed,
        nan=np.finfo(np.float64).min,  # Replace NaNs with minimum float value
        posinf=np.finfo(np.float64).max,  # Replace +inf with maximum float value
        neginf=np.finfo(np.float64).min,  # Replace -inf with minimum float value
    )

    return y_processed


def create_prelim_zip(prelim_output, preprocessing_output):
    """
    Create a ZIP file with all visualization plots and data for download
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save feature and algorithm information
        feat_labels = preprocessing_output.feat_labels
        algo_labels = preprocessing_output.algo_labels

        # Save metadata
        metadata = {
            "total_instances": prelim_output.x.shape[0],
            "total_features": prelim_output.x.shape[1],
            "total_algorithms": prelim_output.y.shape[1],
            "feature_labels": feat_labels,
            "algorithm_labels": algo_labels,
        }

        with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save transformation parameters
        transform_data = []
        for i, feat in enumerate(feat_labels):
            transform_data.append(
                {
                    "Feature": feat,
                    "Box-Cox λ": float(prelim_output.lambda_x[i]),
                    "Mean (μ)": float(prelim_output.mu_x[i]),
                    "Std Dev (σ)": float(prelim_output.sigma_x[i]),
                    "Min Value": float(prelim_output.min_x[i]),
                    "Median": float(prelim_output.med_val[i]),
                    "IQ Range": float(prelim_output.iq_range[i]),
                    "Lower Bound": float(prelim_output.lo_bound[i]),
                    "Upper Bound": float(prelim_output.hi_bound[i]),
                }
            )

        pd.DataFrame(transform_data).to_csv(
            os.path.join(tmp_dir, "feature_parameters.csv"), index=False
        )

        # Save algorithm performance stats
        algo_data = []
        for i, algo in enumerate(algo_labels):
            good_count = np.sum(prelim_output.y_bin[:, i])
            total_count = prelim_output.y_bin.shape[0]
            good_percentage = (good_count / total_count) * 100

            algo_data.append(
                {
                    "Algorithm": algo,
                    "Good Performance Count": int(good_count),
                    "Good Performance (%)": float(good_percentage),
                    "Box-Cox λ": float(prelim_output.lambda_y[i]),
                    "Mean (μ)": float(prelim_output.mu_y[i]),
                    "Std Dev (σ)": float(prelim_output.sigma_y[i]),
                }
            )

        pd.DataFrame(algo_data).to_csv(
            os.path.join(tmp_dir, "algorithm_performance.csv"), index=False
        )

        # Save best algorithm counts
        best_algo_counts = np.bincount(
            prelim_output.p.astype(int), minlength=prelim_output.y.shape[1] + 1
        )[1:]
        best_algo_data = []
        for i, algo in enumerate(algo_labels):
            best_algo_data.append(
                {
                    "Algorithm": algo,
                    "Count": int(best_algo_counts[i]),
                    "Percentage": float(best_algo_counts[i] / sum(best_algo_counts) * 100),
                }
            )

        pd.DataFrame(best_algo_data).to_csv(
            os.path.join(tmp_dir, "best_algorithm_counts.csv"), index=False
        )

        # Create an info file with parameters used
        with open(os.path.join(tmp_dir, "analysis_parameters.txt"), "w") as f:
            f.write(f"Prelim Stage Analysis Parameters\n")
            f.write(f"===============================\n\n")
            if hasattr(prelim_output, "max_perf"):
                f.write(
                    f"Performance Direction: {'Maximize' if prelim_output.max_perf else 'Minimize'}\n"
                )
            if hasattr(prelim_output, "abs_perf"):
                f.write(
                    f"Threshold Type: {'Absolute' if prelim_output.abs_perf else 'Relative (%)'}\n"
                )
            if hasattr(prelim_output, "epsilon"):
                f.write(f"Epsilon Threshold: {prelim_output.epsilon:.2f}\n")
            if hasattr(prelim_output, "beta_threshold"):
                f.write(f"Beta Threshold: {prelim_output.beta_threshold:.2f}\n")
            if hasattr(prelim_output, "bound"):
                f.write(f"Outlier Removal: {'Enabled' if prelim_output.bound else 'Disabled'}\n")
            if hasattr(prelim_output, "norm"):
                f.write(f"Normalization: {'Enabled' if prelim_output.norm else 'Disabled'}\n")
            f.write(f"\nDataset Statistics\n")
            f.write(f"==================\n\n")
            f.write(f"Total Instances: {prelim_output.x.shape[0]}\n")
            f.write(f"Total Features: {prelim_output.x.shape[1]}\n")
            f.write(f"Total Algorithms: {prelim_output.y.shape[1]}\n")

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


def display_outlier_detection(prelim_output, preprocessing_output):
    """
    Visualize outlier detection and bounds
    """
    # Extract data
    x = prelim_output.x
    x_raw = prelim_output.x_raw
    feat_labels = preprocessing_output.feat_labels

    # Select a feature to visualize outlier bounds
    selected_feature = st.selectbox(
        "Select a feature to examine outlier detection",
        options=feat_labels,
        index=0,
        key="outlier_feature_selector",
    )
    feat_idx = feat_labels.index(selected_feature)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw data distribution
    sns.kdeplot(x_raw[:, feat_idx], ax=ax, color="blue", label="Raw Data")

    # Plot processed data distribution
    sns.kdeplot(x[:, feat_idx], ax=ax, color="green", label="After Outlier Handling")

    # Add lines for bounds
    ax.axvline(prelim_output.lo_bound[feat_idx], color="red", linestyle="--", label="Lower Bound")
    ax.axvline(prelim_output.hi_bound[feat_idx], color="red", linestyle="--", label="Upper Bound")

    ax.set_title(f"Distribution of {selected_feature} Before and After Outlier Handling")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Density")
    ax.legend()

    st.pyplot(fig)

    # Download button for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Outlier Plot",
        data=buf,
        file_name=f"outlier_detection_{selected_feature}.png",
        mime="image/png",
        key=f"download_outlier_plot_{selected_feature}",
    )


def display_normalization_impact(prelim_output, preprocessing_output):
    """
    Visualize the impact of normalization on feature distributions
    """
    # Extract data
    x = prelim_output.x
    x_raw = prelim_output.x_raw
    feat_labels = preprocessing_output.feat_labels

    # Select another feature for normalization visualization
    norm_feature = st.selectbox(
        "Select a feature to examine normalization effect",
        options=feat_labels,
        index=min(1, len(feat_labels) - 1),
        key="normalization_feature_selector",
    )
    norm_feat_idx = feat_labels.index(norm_feature)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot raw data
    sns.histplot(x_raw[:, norm_feat_idx], ax=ax1, kde=True, color="blue")
    ax1.set_title(f"Raw {norm_feature}")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")

    # Plot normalized data
    sns.histplot(x[:, norm_feat_idx], ax=ax2, kde=True, color="green")
    ax2.set_title(f"Normalized {norm_feature}")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    st.pyplot(fig)

    # Download button for normalization plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Normalization Plot",
        data=buf,
        file_name=f"normalization_{norm_feature}.png",
        mime="image/png",
        key=f"download_normalization_plot_{norm_feature}",
    )


def display_boxcox_parameters(prelim_output, preprocessing_output):
    """
    Visualize Box-Cox transformation parameters
    """
    # Extract data
    feat_labels = preprocessing_output.feat_labels

    # Display Box-Cox lambda parameters for features
    lambda_df = pd.DataFrame({"Feature": feat_labels, "Lambda": prelim_output.lambda_x})

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x="Feature", y="Lambda", data=lambda_df, ax=ax)
    ax.set_title("Box-Cox Transformation Parameters (Lambda) for Features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.axhline(y=0, color="red", linestyle="--", label="Log Transform (λ=0)")
    ax.axhline(y=1, color="green", linestyle="--", label="No Transform (λ=1)")
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

    # Download button for Box-Cox parameters
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Box-Cox Parameters Plot",
        data=buf,
        file_name="boxcox_parameters.png",
        mime="image/png",
        key="download_boxcox_plot",
    )


def display_feature_correlations(prelim_output, preprocessing_output):
    """
    Display a correlation heatmap of features
    """
    # Extract data and labels
    x = prelim_output.x
    feat_labels = preprocessing_output.feat_labels

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(x, rowvar=False)

    # Create a DataFrame for better visualization
    corr_df = pd.DataFrame(corr_matrix, columns=feat_labels, index=feat_labels)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_df,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()

    st.pyplot(fig)

    # Download button for the correlation matrix
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Correlation Matrix",
        data=buf,
        file_name="feature_correlation_matrix.png",
        mime="image/png",
        key="download_correlation_matrix",
    )


def display_feature_distributions(prelim_output, preprocessing_output):
    """
    Display distribution plots for all features
    """
    # Extract data
    x = prelim_output.x
    feat_labels = preprocessing_output.feat_labels

    # Let user select how many features to display per row
    features_per_row = st.slider(
        "Features per row", min_value=1, max_value=4, value=2, key="features_per_row_slider"
    )

    # Calculate number of rows needed
    num_features = len(feat_labels)
    num_rows = (num_features + features_per_row - 1) // features_per_row

    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, features_per_row, figsize=(15, 3 * num_rows))

    # Flatten axes if it's a 2D array
    if num_rows > 1 and features_per_row > 1:
        axes = axes.flatten()
    elif num_rows == 1 and features_per_row > 1:
        axes = np.array([axes])
    elif num_rows > 1 and features_per_row == 1:
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    # Plot each feature
    for i, feat_label in enumerate(feat_labels):
        if i < len(axes):
            sns.histplot(x[:, i], kde=True, ax=axes[i], color="green")
            axes[i].set_title(f"{feat_label} Distribution")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")

    # Hide any unused subplots
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    # Download button for the distributions
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Feature Distributions",
        data=buf,
        file_name="feature_distributions.png",
        mime="image/png",
        key="download_feature_distributions",
    )


def display_algorithm_comparison(prelim_output, preprocessing_output):
    """
    Display comparative performance metrics for algorithms
    """
    # Extract data
    y_bin = prelim_output.y_bin
    algo_labels = preprocessing_output.algo_labels

    # Calculate metrics
    algo_performance = np.mean(y_bin, axis=0) * 100  # Percentage of instances with good performance
    best_algo_counts = np.bincount(prelim_output.p.astype(int), minlength=len(algo_labels) + 1)[1:]
    best_algo_percentage = best_algo_counts / sum(best_algo_counts) * 100

    # Create DataFrame for visualization
    algo_df = pd.DataFrame(
        {
            "Algorithm": algo_labels,
            "Good Performance (%)": algo_performance,
            "Best Algorithm Count": best_algo_counts,
            "Best Algorithm (%)": best_algo_percentage,
        }
    )

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Good Performance
    bars1 = sns.barplot(
        x="Algorithm", y="Good Performance (%)", data=algo_df, ax=ax1, palette="viridis"
    )
    ax1.set_title("Percentage of Instances with Good Performance")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_ylim(0, 100)

    # Add percentage labels on bars
    for bar, percentage in zip(bars1.patches, algo_performance):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )

    # Plot 2: Best Algorithm
    bars2 = sns.barplot(x="Algorithm", y="Best Algorithm (%)", data=algo_df, ax=ax2, palette="mako")
    ax2.set_title("Percentage of Instances Where Algorithm is Best")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_ylim(0, 100)

    # Add percentage labels on bars
    for bar, percentage in zip(bars2.patches, best_algo_percentage):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    st.pyplot(fig)

    # Display a data table with all metrics
    st.subheader("Algorithm Performance Metrics")
    st.dataframe(
        algo_df.style.format({"Good Performance (%)": "{:.2f}%", "Best Algorithm (%)": "{:.2f}%"})
    )

    # Download button for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Algorithm Comparison",
        data=buf,
        file_name="algorithm_comparison.png",
        mime="image/png",
        key="download_algorithm_comparison",
    )


def display_algorithm_performance_heatmap(prelim_output, preprocessing_output):
    """
    Display a heatmap of algorithm performance across instances
    """
    # Extract data
    y_bin = prelim_output.y_bin
    algo_labels = preprocessing_output.algo_labels

    # Allow user to select a subset of instances to visualize
    max_instances = min(100, y_bin.shape[0])  # Limit to 100 instances for visualization
    num_instances = st.slider(
        "Number of instances to visualize",
        min_value=10,
        max_value=max_instances,
        value=min(50, max_instances),
        key="heatmap_instances_slider",
    )

    # Sample instances if too many
    if y_bin.shape[0] > num_instances:
        sampled_indices = np.random.choice(y_bin.shape[0], num_instances, replace=False)
        y_bin_sample = y_bin[sampled_indices, :]
    else:
        y_bin_sample = y_bin

    # Create instance labels
    instance_labels = [f"Instance {i+1}" for i in range(y_bin_sample.shape[0])]

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(y_bin_sample, columns=algo_labels, index=instance_labels)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, num_instances / 5)))

    sns.heatmap(
        heatmap_df, cmap="YlGnBu", cbar_kws={"label": "Performance (0=Poor, 1=Good)"}, ax=ax
    )

    ax.set_title("Algorithm Performance Across Instances")
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Instances")

    plt.tight_layout()
    st.pyplot(fig)

    # Download button for the heatmap
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Performance Heatmap",
        data=buf,
        file_name="performance_heatmap.png",
        mime="image/png",
        key="download_performance_heatmap",
    )


def display_feature_importance(prelim_output, preprocessing_output):
    """
    Display feature importance based on correlation with algorithm performance
    """
    # Extract data
    x = prelim_output.x
    y_bin = prelim_output.y_bin
    feat_labels = preprocessing_output.feat_labels
    algo_labels = preprocessing_output.algo_labels

    # Calculate correlation between features and algorithm performance
    correlations = []
    for i, algo in enumerate(algo_labels):
        corr_values = []
        for j, feat in enumerate(feat_labels):
            corr = np.abs(np.corrcoef(x[:, j], y_bin[:, i])[0, 1])
            corr_values.append(corr)
        correlations.append(corr_values)

    # Create DataFrame for visualization
    corr_df = pd.DataFrame(correlations, columns=feat_labels, index=algo_labels)

    # Add option to select algorithms
    selected_algos = st.multiselect(
        "Select algorithms to view feature importance",
        options=algo_labels,
        default=[algo_labels[0]] if algo_labels else [],
        key="feature_importance_algo_selector",
    )

    if not selected_algos:
        st.warning("Please select at least one algorithm to view feature importance.")
        return

    # Filter the DataFrame for selected algorithms
    filtered_df = corr_df.loc[selected_algos]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, len(selected_algos) * 0.8 + 2))

    sns.heatmap(
        filtered_df,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Absolute Correlation"},
        ax=ax,
    )

    ax.set_title("Feature Importance (Correlation with Algorithm Performance)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Algorithms")

    plt.tight_layout()
    st.pyplot(fig)

    # For each selected algorithm, show top features
    st.subheader("Top Important Features by Algorithm")

    for algo in selected_algos:
        # Get feature importance for this algorithm
        algo_importances = corr_df.loc[algo].sort_values(ascending=False)

        # Display top 5 features or all if less than 5
        top_n = min(5, len(algo_importances))
        top_features = algo_importances.head(top_n)

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 3))
        top_features.plot(kind="barh", ax=ax)
        ax.set_title(f"Top {top_n} Important Features for {algo}")
        ax.set_xlabel("Absolute Correlation")
        plt.tight_layout()
        st.pyplot(fig)

    # Download button for the heatmap
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Feature Importance",
        data=buf,
        file_name="feature_importance.png",
        mime="image/png",
        key="download_feature_importance",
    )


# ---------- MAIN APPLICATION FUNCTION ---------- #


def show():
    st.header("🔍 Prelim Stage")
    st.write(
        """
    The Prelim stage prepares your data for analysis by cleaning, normalizing, and classifying algorithm performance.
    This stage handles outlier detection, feature normalization, and performance classification.
    """
    )

    # Check if preprocessing output exists
    if not cache_exists("preprocessing_output.pkl"):
        st.error("🚫 Preprocessing output not found. Please run the Preprocessing stage first.")
        if st.button("Go to Preprocessing Stage"):
            st.session_state.current_tab = "preprocessing"
            st.rerun()
        return
    else:
        preprocessing_output = load_from_cache("preprocessing_output.pkl")
        st.success("✅ Preprocessing data loaded successfully!")

    # Configuration form
    with st.form("prelim_config_form"):
        st.subheader("Prelim Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Performance Settings")

            max_perf = st.radio(
                "Performance Objective",
                options=["Maximize", "Minimize"],
                index=0,
                key="performance_objective_radio",
                help="Whether to maximize performance (higher is better) or minimize (lower is better)",
            )

            abs_perf = st.radio(
                "Performance Threshold Type",
                options=["Absolute", "Relative"],
                index=1,  # Default to "Relative" as requested
                key="threshold_type_radio",
                help="Absolute uses fixed values; Relative uses percentages of the best performance",
            )

            epsilon = st.slider(
                "Performance Threshold (epsilon)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Threshold for good performance. If absolute, direct value; if relative, % of best performance",
            )

            beta_threshold = st.slider(
                "Beta Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Fraction of algorithms that must perform well for an instance to be considered 'good'",
            )

        with col2:
            st.subheader("Data Processing Settings")

            bound = st.checkbox(
                "Remove Outliers",
                value=True,
                help="If checked, extreme outliers will be detected and handled",
            )

            norm = st.checkbox(
                "Normalize Data",
                value=True,
                help="If checked, data will be normalized using Box-Cox and Z-score transformations",
            )

            st.subheader("Instance Selection")

            small_scale_flag = st.checkbox(
                "Use Subset of Instances",
                value=False,
                help="If checked, only a fraction of instances will be used",
            )

            small_scale = st.slider(
                "Subset Fraction",
                min_value=0.01,
                max_value=0.99,
                value=0.1,
                step=0.01,
                disabled=not small_scale_flag,
                help="Fraction of instances to use if subset is enabled",
            )

            density_flag = st.checkbox(
                "Use Density-based Filtering",
                value=False,
                help="If checked, instances will be filtered based on density criteria",
            )

            density_method = st.selectbox(
                "Density Method",
                options=["manual", "kmeans", "dbscan"],
                disabled=not density_flag,
                key="density_method_selector",
                help="Method to use for density-based filtering",
            )

            min_distance = st.slider(
                "Minimum Distance",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                disabled=not density_flag,
                help="Minimum distance parameter for density-based filtering",
            )

        submitted = st.form_submit_button("Run Prelim Stage")

    if submitted:
        try:
            # Create options objects
            prelim_opts = PrelimOptions(
                bound=bound,
                norm=norm,
                max_perf=(max_perf == "Maximize"),
                abs_perf=(abs_perf == "Absolute"),
                epsilon=epsilon,
                beta_threshold=beta_threshold,
            )

            selvars_opts = SelvarsOptions(
                feats=None,
                algos=None,
                small_scale_flag=small_scale_flag,
                small_scale=small_scale,
                file_idx_flag=False,
                file_idx="",  # Empty string instead of None
                selvars_type=density_method,
                min_distance=min_distance,
                density_flag=density_flag,
            )

            # Create input object - ensure data is the correct type to avoid overflow errors
            x = preprocessing_output.x.astype(float)
            y = preprocessing_output.y.astype(float)
            x_raw = preprocessing_output.x_raw.astype(float)
            y_raw = preprocessing_output.y_raw.astype(float)

            # Apply preprocessing to handle NaN and infinity values
            y = preprocess_performance_matrix(y)
            y_raw = preprocess_performance_matrix(y_raw)

            # Create PrelimInput object
            prelim_in = PrelimInput(
                x=x,
                y=y,
                x_raw=x_raw,
                y_raw=y_raw,
                s=None,
                inst_labels=preprocessing_output.inst_labels,
                prelim_options=prelim_opts,
                selvars_options=selvars_opts,
            )

            # Run Prelim stage with progress indicator
            with st.spinner("Running Prelim stage... This may take a moment."):
                prelim_output = PrelimStage._run(prelim_in)

            if prelim_output is not None:
                # Save to session state and cache
                st.session_state["prelim_output"] = prelim_output
                st.session_state["ran_prelim"] = True
                save_to_cache(prelim_output, "prelim_output.pkl")

                # Show success message
                st.success("✅ Prelim stage run successfully!")
                st.rerun()  # Rerun to show visualizations

        except Exception as e:
            st.error(f"Error running Prelim stage: {str(e)}")
            st.code(traceback.format_exc())

    # Display results only if the stage has been run successfully
    if st.session_state.get("ran_prelim", False) and cache_exists("prelim_output.pkl"):
        try:
            if "prelim_output" not in st.session_state:
                st.session_state["prelim_output"] = load_from_cache("prelim_output.pkl")

            prelim_output = st.session_state["prelim_output"]

            # Show comprehensive visualizations using tabs
            st.subheader("Analysis Results")

            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Instances", prelim_output.x.shape[0])
            with col2:
                st.metric("Total Features", prelim_output.x.shape[1])
            with col3:
                st.metric("Total Algorithms", prelim_output.y.shape[1])

            # Display parameter choices if available
            if hasattr(prelim_output, "max_perf") and hasattr(prelim_output, "abs_perf"):
                st.subheader("Analysis Parameters")

                params_dict = {
                    "Performance Direction": "Maximize" if prelim_output.max_perf else "Minimize",
                    "Threshold Type": "Absolute" if prelim_output.abs_perf else "Relative (%)",
                    "Epsilon Threshold": f"{prelim_output.epsilon:.2f}",
                    "Beta Threshold": f"{prelim_output.beta_threshold:.2f}",
                    "Outlier Removal": "Enabled" if prelim_output.bound else "Disabled",
                    "Normalization": "Enabled" if prelim_output.norm else "Disabled",
                }

                st.table(pd.DataFrame([params_dict]).T.rename(columns={0: "Value"}))

            # Create tabs for the different visualizations
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "Algorithm Performance",
                    "Feature Distributions",
                    "Feature Correlations",
                    "Outlier Detection",
                    "Normalization Impact",
                    "Box-Cox Parameters",
                ]
            )

            with tab1:
                # Algorithm Performance Analysis
                st.subheader("🎯 Algorithm Performance Analysis")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "This visualization provides a comprehensive comparison of algorithm performance. "
                        "The left chart shows the percentage of instances where each algorithm performs well according to the threshold. "
                        "The right chart shows how often each algorithm is the best-performing algorithm across all instances."
                    )
                display_algorithm_comparison(prelim_output, preprocessing_output)

                # Performance Heatmap
                st.subheader("🌡️ Performance Heatmap")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "This heatmap visualizes the binary performance (good or poor) of each algorithm across a subset of instances. "
                        "Each row represents an instance, and each column represents an algorithm. "
                        "Blue cells indicate good performance, while yellow cells indicate poor performance. "
                        "This helps identify patterns in which algorithms work well on similar instances."
                    )
                display_algorithm_performance_heatmap(prelim_output, preprocessing_output)

                # Feature Importance
                st.subheader("🔍 Feature Importance by Algorithm")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "This visualization shows which features are most strongly correlated with each algorithm's performance. "
                        "Higher values indicate features that have a stronger relationship with algorithm performance, "
                        "suggesting they may be more important for predicting when an algorithm will perform well."
                    )
                display_feature_importance(prelim_output, preprocessing_output)

            with tab2:
                # Feature Distributions
                st.subheader("📊 Feature Distributions")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "This visualization shows the distribution of values for each feature across all instances. "
                        "The distributions shown are after any normalization and outlier handling, "
                        "providing insight into the shape and characteristics of your feature space."
                    )
                display_feature_distributions(prelim_output, preprocessing_output)

            with tab3:
                # Feature Correlations
                st.subheader("🔗 Feature Correlations")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "This heatmap shows the Pearson correlation coefficients between all pairs of features. "
                        "Strong positive correlations (close to 1.0) are shown in warm colors, "
                        "while strong negative correlations (close to -1.0) are shown in cool colors. "
                        "This helps identify redundant features and relationships within your feature space."
                    )
                display_feature_correlations(prelim_output, preprocessing_output)

            with tab4:
                # Outlier detection
                st.subheader("📏 Outlier Detection")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "This visualization shows the distribution of feature values before and after outlier handling. "
                        "The red lines indicate the upper and lower bounds (median ± 5 × IQR) applied to each feature. "
                        "Values outside these bounds are considered outliers and have been adjusted."
                    )
                display_outlier_detection(prelim_output, preprocessing_output)

            with tab5:
                # Normalization impact
                st.subheader("🔄 Normalization Impact")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "This visualization shows the impact of Box-Cox transformation and Z-score normalization on the data. "
                        "The left side shows the raw feature distribution, which may be skewed. "
                        "The right side shows the normalized distribution, which should be more symmetric and standardized."
                    )
                display_normalization_impact(prelim_output, preprocessing_output)

            with tab6:
                # Box-Cox parameters
                st.subheader("📐 Box-Cox Parameters")
                with st.expander("❓ What is this visualization?", expanded=False):
                    st.write(
                        "The Box-Cox transformation parameter (lambda) indicates the power transformation applied to each feature. "
                        "Values near 0 suggest a log transformation was needed, 1 indicates no transformation was needed, "
                        "and values between 0 and 1 indicate different degrees of power transformation."
                    )
                display_boxcox_parameters(prelim_output, preprocessing_output)

            # Add option to download comprehensive results
            st.subheader("Download All Results")

            zip_data = create_prelim_zip(prelim_output, preprocessing_output)

            st.download_button(
                label="📥 Download Complete Analysis (ZIP)",
                data=zip_data,
                file_name="prelim_analysis.zip",
                mime="application/zip",
                help="Download all visualizations, correlation matrices, and parameter information",
                key="download_complete_analysis_zip",
            )

            # Cache management
            st.subheader("🗑️ Cache Management")
            if st.button("❌ Delete Prelim Cache"):
                success = delete_cache("prelim_output.pkl")
                if success:
                    st.success("🗑️ Prelim cache deleted.")
                    if "prelim_output" in st.session_state:
                        del st.session_state["prelim_output"]
                    if "ran_prelim" in st.session_state:
                        del st.session_state["ran_prelim"]
                    st.rerun()
                else:
                    st.warning("⚠️ No cache file found to delete.")

        except Exception as e:
            st.error(f"Error displaying Prelim results: {str(e)}")
            st.code(traceback.format_exc())
    else:
        if cache_exists("prelim_output.pkl") and not st.session_state.get("ran_prelim", False):
            st.session_state["ran_prelim"] = True
            st.rerun()


if __name__ == "__main__":
    show()
