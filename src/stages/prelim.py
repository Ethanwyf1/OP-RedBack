import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback
import io
import zipfile
from sklearn.decomposition import PCA

from instancespace.data.options import PrelimOptions, SelvarsOptions
from instancespace.stages.prelim import PrelimStage, PrelimOutput
from utils.download_utils import create_stage_output_zip
from utils.cache_utils import save_to_cache, load_from_cache, delete_cache, cache_exists


class ExtendedPrelimOutput:
    """Wrapper for PrelimOutput that allows adding custom attributes."""
    
    def __init__(self, prelim_output, max_perf=True, abs_perf=True, epsilon=0.1, beta_threshold=0.1, bound=True, norm=True):
        # Copy all attributes from the PrelimOutput
        for attr in dir(prelim_output):
            if not attr.startswith('_'):  # Skip private attributes
                setattr(self, attr, getattr(prelim_output, attr))
        
        # Add custom attributes
        self.max_perf = max_perf
        self.abs_perf = abs_perf
        self.epsilon = epsilon
        self.beta_threshold = beta_threshold
        self.bound = bound
        self.norm = norm


def custom_prelim_run(x, y, x_raw, y_raw, s, inst_labels, prelim_options, selvars_options):
    """
    A custom wrapper for the Prelim stage to handle type issues with infinity values.
    This removes NaN rows to avoid conversion issues and performs the preprocessing.
    
    Returns a safely constructed PrelimOutput object.
    """
    # Remove rows with NaN values in either x or y
    nan_mask_x = np.isnan(x).any(axis=1)
    nan_mask_y = np.isnan(y).any(axis=1)
    nan_mask = nan_mask_x | nan_mask_y
    
    if nan_mask.any():
        st.warning(f"Removing {np.sum(nan_mask)} rows with NaN values")
        valid_rows = ~nan_mask
        x = x[valid_rows]
        y = y[valid_rows]
        x_raw = x_raw[valid_rows]
        y_raw = y_raw[valid_rows]
        
        if inst_labels is not None:
            inst_labels = inst_labels[valid_rows]
        if s is not None:
            s = s[valid_rows]
    
    # Ensure data is float type to handle any calculations
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_raw = x_raw.astype(np.float64)
    y_raw = y_raw.astype(np.float64)
    
    # Create binary performance evaluation
    y_bin = np.zeros_like(y, dtype=bool)
    
    # Binary classification based on performance
    if prelim_options.max_perf:
        # For maximization problems
        if prelim_options.abs_perf:
            # Absolute threshold
            y_bin = y >= prelim_options.epsilon
        else:
            # Relative to best
            y_best = np.max(y, axis=1, keepdims=True)
            y_bin = y >= (y_best * (1 - prelim_options.epsilon))
    else:
        # For minimization problems
        if prelim_options.abs_perf:
            # Absolute threshold
            y_bin = y <= prelim_options.epsilon
        else:
            # Relative to best
            y_best = np.min(y, axis=1, keepdims=True)
            y_bin = y <= (y_best * (1 + prelim_options.epsilon))
    
    # Calculate y_best as a flat array - best performance per instance
    if prelim_options.max_perf:
        y_best = np.max(y, axis=1)
    else:
        y_best = np.min(y, axis=1)
    
    # Calculate p - index of best algorithm per instance (1-indexed)
    if prelim_options.max_perf:
        p = np.argmax(y, axis=1) + 1
    else:
        p = np.argmin(y, axis=1) + 1
    
    # Calculate num_good_algos - number of good algorithms per instance
    num_good_algos = np.sum(y_bin, axis=1)
    
    # Calculate beta - instances with more than threshold*nalgos good algorithms
    beta = num_good_algos > (prelim_options.beta_threshold * y.shape[1])
    
    # Create placeholders for other required values
    nfeats = x.shape[1]
    nalgos = y.shape[1]
    
    # Handle outliers if requested
    if prelim_options.bound:
        # Calculate bounds for outlier removal
        med_val = np.median(x, axis=0)
        iq_range = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        hi_bound = med_val + 5 * iq_range
        lo_bound = med_val - 5 * iq_range
        
        # Apply bounds
        for i in range(nfeats):
            x[:, i] = np.clip(x[:, i], lo_bound[i], hi_bound[i])
    else:
        # Default values if not calculating bounds
        med_val = np.median(x, axis=0)
        iq_range = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        hi_bound = med_val + 5 * iq_range
        lo_bound = med_val - 5 * iq_range
    
    # Handle normalization if requested
    if prelim_options.norm:
        # For features (x)
        min_x = np.min(x, axis=0)
        x = x - min_x  # Center around 0
        
        # Simple z-score normalization instead of Box-Cox
        mu_x = np.mean(x, axis=0)
        sigma_x = np.std(x, axis=0)
        for i in range(nfeats):
            if sigma_x[i] > 0:  # Avoid division by zero
                x[:, i] = (x[:, i] - mu_x[i]) / sigma_x[i]
        
        # For algorithms (y)
        min_y = float(np.min(y))
        y = y - min_y  # Make all values positive
        
        mu_y = np.mean(y, axis=0)
        sigma_y = np.std(y, axis=0)
        for i in range(nalgos):
            if sigma_y[i] > 0:  # Avoid division by zero
                y[:, i] = (y[:, i] - mu_y[i]) / sigma_y[i]
        
        # Placeholder for lambda values (normally from Box-Cox)
        lambda_x = np.zeros(nfeats)
        lambda_y = np.zeros(nalgos)
    else:
        # Default values if not normalizing
        min_x = np.min(x, axis=0)
        lambda_x = np.zeros(nfeats)
        mu_x = np.mean(x, axis=0)
        sigma_x = np.std(x, axis=0)
        
        min_y = float(np.min(y))
        lambda_y = np.zeros(nalgos)
        mu_y = np.mean(y, axis=0)
        sigma_y = np.std(y, axis=0)
    
    # Create a PrelimOutput object with our calculated values
    prelim_output = PrelimOutput(
        med_val=med_val,
        iq_range=iq_range,
        hi_bound=hi_bound,
        lo_bound=lo_bound,
        min_x=min_x,
        lambda_x=lambda_x,
        mu_x=mu_x,
        sigma_x=sigma_x,
        min_y=min_y,
        lambda_y=lambda_y,
        sigma_y=sigma_y,
        mu_y=mu_y,
        x=x,
        y=y,
        x_raw=x_raw,
        y_raw=y_raw,
        y_bin=y_bin,
        y_best=y_best,
        p=p.astype(np.int_),
        num_good_algos=num_good_algos,
        beta=beta,
        instlabels=inst_labels,
        data_dense=None,
        s=s
    )
    
    # Wrap the PrelimOutput in our ExtendedPrelimOutput to add custom attributes
    extended_output = ExtendedPrelimOutput(
        prelim_output,
        max_perf=prelim_options.max_perf,
        abs_perf=prelim_options.abs_perf,
        epsilon=prelim_options.epsilon,
        beta_threshold=prelim_options.beta_threshold,
        bound=prelim_options.bound,
        norm=prelim_options.norm
    )
    
    return extended_output


def run_prelim(preprocessing_output, prelim_options, selvars_options):
    """
    Run the Prelim stage of ISA using the output from Preprocessing stage.
    This function uses a custom implementation to avoid infinity conversion issues.
    """
    try:
        # Use our custom implementation to avoid the infinity conversion issue
        return custom_prelim_run(
            x=preprocessing_output.x.copy(),
            y=preprocessing_output.y.copy(),
            x_raw=preprocessing_output.x_raw.copy(),
            y_raw=preprocessing_output.y_raw.copy(),
            s=preprocessing_output.s,
            inst_labels=preprocessing_output.inst_labels,
            prelim_options=prelim_options,
            selvars_options=selvars_options
        )
    except Exception as e:
        st.error(f"Error during Prelim execution: {str(e)}")
        st.code(traceback.format_exc())
        return None


def viz_1_algorithm_performance_comparison(output):
    """
    Creates a violin plot comparing algorithm performance distributions.
    This visualization shows both raw performance and classification results.
    """
    # Prepare the data
    algo_labels = [f"Algorithm {i+1}" for i in range(output.y.shape[1])]
    
    # Reshape data for plotting
    perf_data = []
    for i in range(output.y_raw.shape[1]):
        algo_perf = output.y_raw[:, i]
        algo_bin = output.y_bin[:, i]
        
        for j, (perf, is_good) in enumerate(zip(algo_perf, algo_bin)):
            perf_data.append({
                'Algorithm': algo_labels[i],
                'Performance': perf,
                'Classification': 'Good' if is_good else 'Poor',
                'Instance': j
            })
    
    # Create a dataframe for plotting
    perf_df = pd.DataFrame(perf_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Violin plot showing performance distribution by algorithm
    sns.violinplot(x='Algorithm', y='Performance', hue='Classification', 
                 data=perf_df, palette={'Good': 'green', 'Poor': 'red'}, 
                 inner='quartile', split=True, ax=ax)
    
    # Customize the plot
    ax.set_title("Algorithm Performance Distribution and Classification")
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Performance Value")
    plt.xticks(rotation=45, ha="right")
    
    # Add a line for absolute threshold if applicable
    if hasattr(output, 'abs_perf') and output.abs_perf and hasattr(output, 'epsilon'):
        ax.axhline(y=output.epsilon, color='blue', linestyle='--', 
                  label=f'Threshold (ε={output.epsilon:.2f})')
        ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.caption("ℹ️ This violin plot shows the performance distribution for each algorithm. "
              "The green side shows instances classified as 'good' performance, while "
              "the red side shows instances classified as 'poor' performance. "
              "The split violin shape helps visualize how performance values are distributed.")
    
    # Additional statistics
    col1, col2 = st.columns(2)
    with col1:
        # Create a bar chart of percent good performance by algorithm
        good_percent = []
        for i in range(output.y_bin.shape[1]):
            pct = np.mean(output.y_bin[:, i]) * 100
            good_percent.append({
                'Algorithm': algo_labels[i],
                'Good Performance (%)': pct
            })
        
        good_df = pd.DataFrame(good_percent)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(good_df['Algorithm'], good_df['Good Performance (%)'], color='skyblue')
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
            
        ax.set_title("Percentage of Instances with Good Performance by Algorithm")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Good Performance (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Show raw performance statistics
        stats = []
        for i in range(output.y_raw.shape[1]):
            stats.append({
                'Algorithm': algo_labels[i],
                'Min': np.min(output.y_raw[:, i]),
                'Max': np.max(output.y_raw[:, i]),
                'Mean': np.mean(output.y_raw[:, i]),
                'Median': np.median(output.y_raw[:, i]),
                'Good Count': np.sum(output.y_bin[:, i])
            })
        
        stats_df = pd.DataFrame(stats)
        st.write("Algorithm Performance Statistics")
        st.dataframe(stats_df, use_container_width=True)


def viz_2_feature_algorithm_relationship(output):
    """
    Creates scatter plots to show relationships between features and algorithm performance.
    This visualization helps understand how features affect algorithm classification.
    """
    # Pick a reasonable number of features and algorithms
    max_features = min(3, output.x.shape[1])
    max_algos = min(3, output.y.shape[1])
    
    # Feature and algorithm labels
    feat_labels = [f"Feature {i+1}" for i in range(max_features)]
    algo_labels = [f"Algorithm {i+1}" for i in range(max_algos)]
    
    st.write("### Feature-Algorithm Relationship Analysis")
    st.write("Select a feature and algorithm to visualize their relationship:")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_feat = st.selectbox("Select Feature", range(max_features), format_func=lambda x: feat_labels[x])
    with col2:
        selected_algo = st.selectbox("Select Algorithm", range(max_algos), format_func=lambda x: algo_labels[x])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the data
    feature_values = output.x[:, selected_feat]
    algo_performance = output.y_raw[:, selected_algo]
    is_good = output.y_bin[:, selected_algo]
    
    # Create scatter plot
    ax.scatter(
        feature_values[is_good], 
        algo_performance[is_good], 
        color='green', 
        label='Good Performance',
        alpha=0.7
    )
    ax.scatter(
        feature_values[~is_good], 
        algo_performance[~is_good], 
        color='red', 
        label='Poor Performance',
        alpha=0.7
    )
    
    # Add a threshold line if applicable
    if hasattr(output, 'abs_perf') and output.abs_perf and hasattr(output, 'epsilon'):
        ax.axhline(y=output.epsilon, color='blue', linestyle='--', 
                  label=f'Threshold (ε={output.epsilon:.2f})')
    
    ax.set_title(f"Relationship between {feat_labels[selected_feat]} and {algo_labels[selected_algo]} Performance")
    ax.set_xlabel(feat_labels[selected_feat])
    ax.set_ylabel(f"{algo_labels[selected_algo]} Performance")
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional feature-performance correlation analysis
    st.write("### Feature-Performance Correlation Analysis")
    
    # Calculate correlations between features and algorithm performance
    correlations = []
    for i in range(max_features):
        for j in range(max_algos):
            feature_vals = output.x[:, i]
            algo_perf = output.y_raw[:, j]
            
            # Calculate correlation coefficient, handle NaN values
            mask = ~np.isnan(feature_vals) & ~np.isnan(algo_perf)
            if np.sum(mask) > 1:  # Need at least 2 points for correlation
                corr = np.corrcoef(feature_vals[mask], algo_perf[mask])[0, 1]
            else:
                corr = np.nan
                
            correlations.append({
                'Feature': feat_labels[i],
                'Algorithm': algo_labels[j],
                'Correlation': corr
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # Reshape for heatmap
    corr_pivot = corr_df.pivot(index='Feature', columns='Algorithm', values='Correlation')
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr_pivot, 
        cmap='coolwarm', 
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title("Feature-Algorithm Performance Correlation")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.caption("ℹ️ This analysis shows how features relate to algorithm performance. "
              "Strong positive correlations (close to 1.0) indicate that as the feature value increases, "
              "algorithm performance tends to improve. Strong negative correlations (close to -1.0) "
              "indicate that as the feature value increases, algorithm performance tends to worsen.")


def viz_3_multi_algorithm_instances(output):
    """
    Visualizes instances where multiple algorithms perform well (beta=True).
    
    This helps understand the beta calculation in the Prelim stage.
    """
    # Calculate number of good algorithms per instance
    num_good_algos = output.num_good_algos
    total_algos = output.y_bin.shape[1]
    
    # Get beta value (instances with multiple good algorithms)
    beta = output.beta
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram of number of good algorithms per instance
    bins = np.arange(0, total_algos + 2) - 0.5  # Bins centered on integers
    ax1.hist(num_good_algos, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add beta threshold line if we have that information
    if hasattr(output, 'beta_threshold'):
        beta_threshold = output.beta_threshold * total_algos
        ax1.axvline(x=beta_threshold, color='red', linestyle='--', label=f'β Threshold ({beta_threshold:.1f})')
        ax1.legend()
    
    ax1.set_title("Distribution of Good Algorithms per Instance")
    ax1.set_xlabel("Number of Good Algorithms")
    ax1.set_ylabel("Number of Instances")
    ax1.set_xticks(range(total_algos + 1))
    
    # Pie chart of beta classification
    beta_counts = [np.sum(~beta), np.sum(beta)]
    ax2.pie(
        beta_counts, 
        labels=['Single Best Algorithm', 'Multiple Good Algorithms (β=True)'],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff'],
        startangle=90
    )
    ax2.set_title("Instance Classification by β")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.caption("ℹ️ This visualization shows the distribution of instances by the number of algorithms that perform well on them. "
              "The left chart shows how many instances have 0, 1, 2, etc. good algorithms. "
              "The right chart shows the proportion of instances where multiple algorithms perform well (β=True) "
              "versus instances with a single best algorithm.")
    
    # Feature analysis for instances with many good algorithms
    st.write("### Feature Analysis for Beta Classification")
    
    if np.sum(beta) > 0 and np.sum(~beta) > 0:
        # Calculate feature means for beta=True vs beta=False
        feat_labels = [f"Feature {i+1}" for i in range(output.x.shape[1])]
        feature_stats = []
        
        for i in range(output.x.shape[1]):
            feature_vals = output.x[:, i]
            
            beta_mean = np.mean(feature_vals[beta])
            non_beta_mean = np.mean(feature_vals[~beta])
            
            feature_stats.append({
                'Feature': feat_labels[i],
                'β=True Mean': beta_mean,
                'β=False Mean': non_beta_mean,
                'Difference': beta_mean - non_beta_mean,
                'Ratio': beta_mean / non_beta_mean if non_beta_mean != 0 else float('nan')
            })
        
        stats_df = pd.DataFrame(feature_stats)
        
        # Sort by absolute difference to find most discriminative features
        stats_df['Abs Difference'] = np.abs(stats_df['Difference'])
        stats_df = stats_df.sort_values('Abs Difference', ascending=False).drop('Abs Difference', axis=1)
        
        st.write("Feature statistics by beta classification:")
        st.dataframe(stats_df, use_container_width=True)
        
        # Plot top discriminative features
        top_n = min(3, len(feat_labels))
        top_features = stats_df.iloc[:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(top_n)
        width = 0.35
        
        ax.bar(x - width/2, top_features['β=True Mean'], width, label='β=True (Multiple Good Algorithms)')
        ax.bar(x + width/2, top_features['β=False Mean'], width, label='β=False (Single Best Algorithm)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(top_features['Feature'], rotation=45, ha='right')
        ax.set_title("Feature Comparison by Beta Classification")
        ax.set_ylabel("Feature Mean Value")
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("This chart compares feature values between instances with multiple good algorithms (β=True) "
                  "and instances with a single best algorithm (β=False). "
                  "Large differences suggest that these features strongly influence whether multiple algorithms perform well.")
    else:
        st.info("Not enough data to compare features between beta groups. Need instances in both β=True and β=False categories.")


def show():
    st.header("🔬 Prelim Stage")
    
    # Check if preprocessing has been run
    if not cache_exists("preprocessing_output.pkl"):
        st.error("🚫 Preprocessing output not found. Please run the Preprocessing stage first.")
        if st.button("Go to Preprocessing Stage"):
            st.session_state.current_tab = "preprocessing"
            st.experimental_rerun()
        return
    else:
        preprocessing_output = load_from_cache("preprocessing_output.pkl")
        st.success("✅ Preprocessing data loaded successfully!")
    
    # Configuration Section
    st.subheader("⚙️ Configure Prelim Options")
    
    with st.form("prelim_config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Settings")
            max_perf = st.checkbox("Maximize Performance", value=True, 
                                help="If checked, higher values are better. If unchecked, lower values are better.")
            abs_perf = st.checkbox("Absolute Performance", value=True,
                                 help="If checked, use absolute performance threshold. If unchecked, use relative to best.")
            epsilon = st.slider("Performance Threshold (ε)", 
                              min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                              help="Performance threshold for determining good algorithms")
            beta_threshold = st.slider("Beta Threshold", 
                                    min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                                    help="Threshold for determining good instances")
        
        with col2:
            st.subheader("Data Transformation")
            bound = st.checkbox("Remove Outliers", value=True, 
                              help="If checked, outliers will be removed")
            norm = st.checkbox("Normalize Data", value=True,
                             help="If checked, data will be normalized")
            
            small_scale_flag = st.checkbox("Use Small-Scale Experiment", value=False)
            small_scale = st.slider("Small-Scale Percentage", 
                                  min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                                  disabled=not small_scale_flag)
            
            density_flag = st.checkbox("Filter by Density", value=False)
            min_distance = st.slider("Minimum Distance", 
                                   min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                   disabled=not density_flag)
        
        submit_button = st.form_submit_button("Run Prelim Stage")
    
    if submit_button:
        try:
            with st.spinner("Running Prelim stage..."):
                # Create options for Prelim
                prelim_options = PrelimOptions(
                    max_perf=max_perf,
                    abs_perf=abs_perf,
                    epsilon=epsilon,
                    beta_threshold=beta_threshold,
                    bound=bound,
                    norm=norm
                )
                
                selvars_options = SelvarsOptions(
                    feats=None,  # Using all features from preprocessing stage
                    algos=None,  # Using all algorithms from preprocessing stage
                    small_scale_flag=small_scale_flag,
                    small_scale=small_scale,
                    file_idx_flag=False,
                    file_idx=None,
                    selvars_type="manual" if not density_flag else "density",
                    min_distance=min_distance if density_flag else 0.0,
                    density_flag=density_flag
                )
                
                # Run the Prelim stage with our custom implementation
                prelim_output = run_prelim(preprocessing_output, prelim_options, selvars_options)
                
                if prelim_output is not None:
                    # Save to session state and cache
                    st.session_state["prelim_output"] = prelim_output
                    st.session_state["ran_prelim"] = True
                    save_to_cache(prelim_output, "prelim_output.pkl")
                    
                    # Show success message
                    st.toast("✅ Prelim stage run successfully!", icon="🚀")
        except Exception as e:
            st.error(f"Error running Prelim stage: {str(e)}")
            st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.get("ran_prelim", False) or cache_exists("prelim_output.pkl"):
        try:
            if "prelim_output" not in st.session_state and cache_exists("prelim_output.pkl"):
                st.session_state["prelim_output"] = load_from_cache("prelim_output.pkl")
                st.session_state["ran_prelim"] = True
            
            output = st.session_state["prelim_output"]
            
            # Summary Section
            st.subheader("📊 Summary")
            
            # Create a safer summary that checks for attribute existence
            summary_data = {
                "Metric": [
                    "Number of Instances",
                    "Number of Features",
                    "Number of Algorithms",
                    "Good Algorithms (avg per instance)",
                    "Good Instances (beta=True)",
                    "Optimization Direction",
                    "Performance Threshold Type", 
                    "Epsilon Value",
                    "Beta Threshold"
                ],
                "Value": [
                    output.x.shape[0],
                    output.x.shape[1],
                    output.y.shape[1],
                    round(float(np.mean(output.num_good_algos)), 2),
                    int(np.sum(output.beta)),
                    "Maximize" if hasattr(output, "max_perf") and output.max_perf else "Minimize",
                    "Absolute" if hasattr(output, "abs_perf") and output.abs_perf else "Relative",
                    float(output.epsilon) if hasattr(output, "epsilon") else 0.1,
                    float(output.beta_threshold) if hasattr(output, "beta_threshold") else 0.1
                ],
                "Description": [
                    "Total number of problem instances after filtering",
                    "Number of features selected for analysis",
                    "Number of algorithms in comparison",
                    "Average number of algorithms that performed well per instance",
                    "Number of instances where multiple algorithms perform well",
                    "Whether higher (maximize) or lower (minimize) values are better",
                    "Whether threshold is absolute or relative to best performance",
                    "Threshold used to determine good performance (epsilon)",
                    "Threshold for determining instances with multiple good algorithms"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Visualization tabs
            viz_tabs = st.tabs([
                "Algorithm Performance", 
                "Feature-Algorithm Relationships", 
                "Beta Analysis"
            ])
            
            with viz_tabs[0]:
                viz_1_algorithm_performance_comparison(output)
                
            with viz_tabs[1]:
                viz_2_feature_algorithm_relationship(output)
                
            with viz_tabs[2]:
                viz_3_multi_algorithm_instances(output)
            
            st.success("✅ Prelim stage completed and visualized.")
            
            # Download Button
            st.subheader("📥 Download Prelim Results")
            
            if cache_exists("prelim_output.pkl"):
                cached_output = load_from_cache("prelim_output.pkl")
                
                # Create dataframes from outputs
                try:
                    features_df = pd.DataFrame(
                        cached_output.x, 
                        columns=[f"Feature {i+1}" for i in range(cached_output.x.shape[1])]
                    )
                    performance_df = pd.DataFrame(
                        cached_output.y, 
                        columns=[f"Algo {i+1}" for i in range(cached_output.y.shape[1])]
                    )
                    
                    # Save binary performance to a temporary file
                    binary_df = pd.DataFrame(
                        cached_output.y_bin, 
                        columns=[f"Algo {i+1}" for i in range(cached_output.y_bin.shape[1])]
                    )
                    tmp_dir = "temp_files"
                    os.makedirs(tmp_dir, exist_ok=True)
                    binary_path = os.path.join(tmp_dir, "binary_performance.csv")
                    binary_df.to_csv(binary_path, index=False)
                    
                    # Create the zip with standard parameters
                    zip_data = create_stage_output_zip(
                        x=features_df,
                        y=performance_df,
                        instance_labels=cached_output.instlabels if hasattr(cached_output, 'instlabels') else None,
                        source_labels=cached_output.s if hasattr(cached_output, 's') else None,
                        metadata_description="Processed features and algorithm performance from Prelim stage."
                    )
                    
                    # Manual approach to add binary performance to the zip
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "a") as new_zip:
                        # Read the existing zip file content
                        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as old_zip:
                            for item in old_zip.infolist():
                                data = old_zip.read(item.filename)
                                new_zip.writestr(item, data)
                        
                        # Add binary_performance.csv to the zip
                        new_zip.write(binary_path, arcname="binary_performance.csv")
                    
                    # Get the bytes from the in-memory zip file
                    buffer.seek(0)
                    zip_data_with_binary = buffer.getvalue()
                    
                    # Clean up temporary file
                    if os.path.exists(binary_path):
                        os.remove(binary_path)
                    
                    st.download_button(
                        label="⬇️ Download Prelim Output (ZIP)",
                        data=zip_data_with_binary,
                        file_name="prelim_output.zip",
                        mime="application/zip"
                    )
                except Exception as e:
                    st.error(f"Error creating download package: {str(e)}")
                    st.code(traceback.format_exc())
            else:
                st.warning("⚠️ No prelim cache found. Please click **Run Prelim Stage** first.")
            
            # Delete cache Button
            st.subheader("🗑️ Cache Management")
            
            if st.button("❌ Delete Prelim Cache"):
                success = delete_cache("prelim_output.pkl")
                if success:
                    st.success("🗑️ Prelim cache deleted.")
                    if "prelim_output" in st.session_state:
                        del st.session_state["prelim_output"]
                    if "ran_prelim" in st.session_state:
                        del st.session_state["ran_prelim"]
                else:
                    st.warning("⚠️ No cache file found to delete.")
        
        except Exception as e:
            st.error(f"Error displaying Prelim results: {str(e)}")
            st.code(traceback.format_exc())