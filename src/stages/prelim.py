# Standard Library Imports
import pickle
import streamlit as st
import traceback
import os
import io
import json
import zipfile
import tempfile
from typing import NamedTuple, Optional
from numpy.typing import NDArray

# Data Manipulation and Scientific Computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning and Scientific Libraries
from scipy import optimize, stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Instancespace Specific Imports
from instancespace.data.model import DataDense
from instancespace.data.options import PrelimOptions, SelvarsOptions
from instancespace.stages.stage import Stage
from instancespace.stages.prelim import PrelimStage
from instancespace.utils.filter import do_filter

# Cache Utilities
from utils.cache_utils import load_from_cache, cache_exists, save_to_cache

# Define PrelimInput
class PrelimInput(NamedTuple):
    """
    Input structure for the Preliminary Stage of Instancespace Analysis.
    """
    x: NDArray[np.double]
    y: NDArray[np.double]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]
    s: Optional[pd.Series]
    inst_labels: pd.Series
    prelim_options: PrelimOptions
    selvars_options: SelvarsOptions

def manually_save_output(output, filename="prelim_output.pkl"):
    """
    Custom function to save the preliminary output directly to disk,
    avoiding issues with os.path.join and complex objects.
    """
    try:
        # Define the cache directory
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create the full path
        filepath = os.path.join(cache_dir, filename)
        
        # Save the object using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(output, f)
        
        return True
    except Exception as e:
        st.error(f"Error saving output: {e}")
        return False

# Preprocessing and Type Handling
def preprocess_performance_matrix(y):
    """
    Preprocess the performance matrix to ensure type compatibility.
    """
    # Convert to float64
    y_processed = y.astype(np.float64)
    
    # Replace any NaN or infinity values
    y_processed = np.nan_to_num(
        y_processed, 
        nan=np.finfo(np.float64).min,  # Replace NaNs with minimum float value
        posinf=np.finfo(np.float64).max,  # Replace +inf with maximum float value
        neginf=np.finfo(np.float64).min   # Replace -inf with minimum float value
    )
    
    return y_processed

# Cache Loading
def load_preprocessing_output():
    """
    Load preprocessing output from cache with comprehensive error handling.
    """
    try:
        # Check if preprocessing cache exists
        if not cache_exists("preprocessing_output.pkl"):
            st.error("Preprocessing output cache not found.")
            return None
        
        # Attempt to load preprocessing output from cache
        preprocessing_output = load_from_cache("preprocessing_output.pkl")
        
        # Validate the loaded output
        if preprocessing_output is None:
            st.error("Loaded preprocessing output is None")
            return None
        
        # Verify essential attributes
        essential_attrs = ['x', 'y', 'x_raw', 'y_raw', 'inst_labels']
        missing_attrs = [attr for attr in essential_attrs if not hasattr(preprocessing_output, attr)]
        
        if missing_attrs:
            st.error(f"Missing essential preprocessing output attributes: {', '.join(missing_attrs)}")
            return None
        
        # Log successful loading
        st.success("Preprocessing output loaded successfully")
        
        return preprocessing_output
    
    except Exception as e:
        st.error(f"Error loading preprocessing output: {str(e)}")
        return None

# Extract algorithm and feature names from preprocessing output or zip file
def extract_names_from_preprocessing(preprocessing_output):
    """
    Extract algorithm and feature names directly from the preprocessing output.
    Tries multiple possible attribute names where these might be stored.
    """
    algorithm_names = []
    feature_names = []
    
    try:
        # First check if the names are directly available in the preprocessing_output
        if hasattr(preprocessing_output, 'algorithm_names') and preprocessing_output.algorithm_names is not None:
            algorithm_names = preprocessing_output.algorithm_names
        elif hasattr(preprocessing_output, 'algo_names') and preprocessing_output.algo_names is not None:
            algorithm_names = preprocessing_output.algo_names
        
        if hasattr(preprocessing_output, 'feature_names') and preprocessing_output.feature_names is not None:
            feature_names = preprocessing_output.feature_names
        elif hasattr(preprocessing_output, 'feat_names') and preprocessing_output.feat_names is not None:
            feature_names = preprocessing_output.feat_names
        
        # If we still don't have names, check for DataFrame headers if available
        if not algorithm_names and hasattr(preprocessing_output, 'y_df') and preprocessing_output.y_df is not None:
            # Extract algorithm names from column headers that start with 'algo_'
            algo_cols = [col for col in preprocessing_output.y_df.columns if col.startswith('algo_')]
            if algo_cols:
                algorithm_names = [col.replace('algo_', '') for col in algo_cols]
        
        if not feature_names and hasattr(preprocessing_output, 'x_df') and preprocessing_output.x_df is not None:
            # Extract feature names from column headers that start with 'feature_'
            feat_cols = [col for col in preprocessing_output.x_df.columns if col.startswith('feature_')]
            if feat_cols:
                feature_names = [col.replace('feature_', '') for col in feat_cols]
        
        # If we still don't have names, look for dataframes with correct column prefixes
        if hasattr(preprocessing_output, 'dataframes') and preprocessing_output.dataframes is not None:
            for df_name, df in preprocessing_output.dataframes.items():
                # Look for algorithm columns (algo_*)
                if not algorithm_names:
                    algo_cols = [col for col in df.columns if col.startswith('algo_')]
                    if algo_cols:
                        algorithm_names = [col.replace('algo_', '') for col in algo_cols]
                
                # Look for feature columns (feature_*)
                if not feature_names:
                    feat_cols = [col for col in df.columns if col.startswith('feature_')]
                    if feat_cols:
                        feature_names = [col.replace('feature_', '') for col in feat_cols]
        
        # If we still don't have names, check if there are any raw CSV filenames stored
        if (not algorithm_names or not feature_names) and hasattr(preprocessing_output, 'input_files'):
            # Check if we have a performance.csv or features.csv filename
            for filename in preprocessing_output.input_files:
                if 'performance' in filename.lower() and os.path.exists(filename):
                    try:
                        perf_df = pd.read_csv(filename)
                        algo_cols = [col for col in perf_df.columns if col.startswith('algo_')]
                        if algo_cols:
                            algorithm_names = [col.replace('algo_', '') for col in algo_cols]
                    except:
                        pass
                
                if 'feature' in filename.lower() and os.path.exists(filename):
                    try:
                        feat_df = pd.read_csv(filename)
                        feat_cols = [col for col in feat_df.columns if col.startswith('feature_')]
                        if feat_cols:
                            feature_names = [col.replace('feature_', '') for col in feat_cols]
                    except:
                        pass
        
        # If we still don't have names, use the default column names for the known CSVs from user input
        if not algorithm_names:
            known_algo_names = ['RandomGreedy', 'DSATUR', 'Bktr', 'HillClimber', 'HEA', 
                               'PartialCol', 'TabuCol', 'AntCol']
            
            # Check if the shape matches the known list
            if preprocessing_output.y.shape[1] == len(known_algo_names):
                algorithm_names = known_algo_names
        
        if not feature_names:
            known_feature_names = ['Density', 'AlgConnectivity', 'Energy']
            # Check if the shape matches the known list
            if preprocessing_output.x.shape[1] == len(known_feature_names):
                feature_names = known_feature_names
    
    except Exception as e:
        st.warning(f"Error extracting names from preprocessing output: {str(e)}")
    
    # If we still don't have names, use generic names
    if not algorithm_names:
        algorithm_names = [f"Algorithm {i+1}" for i in range(preprocessing_output.y.shape[1])]
    
    if not feature_names:
        feature_names = [f"Feature {i+1}" for i in range(preprocessing_output.x.shape[1])]
    
    return algorithm_names, feature_names

# Preliminary Stage Processing
def run_prelim(preprocessing_output, prelim_options, selvars_options):
    """
    Run the Prelim stage with enhanced type handling and selvars_options fix.
    """
    try:
        # Preprocess matrices to ensure type compatibility
        x = preprocessing_output.x.astype(np.float64)
        x_raw = preprocessing_output.x_raw.astype(np.float64)
        
        # Special handling for performance matrices
        y = preprocess_performance_matrix(preprocessing_output.y)
        y_raw = preprocess_performance_matrix(preprocessing_output.y_raw)
        
        # Modify selvars_options to provide a default file_idx if None
        if selvars_options.file_idx is None:
            # Create a temporary file with a placeholder path
            temp_file = tempfile.mktemp(suffix='.txt')
            
            # Modify selvars_options with the temporary file path
            selvars_options = SelvarsOptions(
                feats=selvars_options.feats,
                algos=selvars_options.algos,
                small_scale_flag=selvars_options.small_scale_flag,
                small_scale=selvars_options.small_scale,
                file_idx_flag=False,
                file_idx=temp_file,
                selvars_type=selvars_options.selvars_type,
                min_distance=selvars_options.min_distance,
                density_flag=selvars_options.density_flag
            )
        
        # Perform preliminary processing
        st.info("Running preliminary processing...")
        prelim_output = PrelimStage._run(
            PrelimInput(
                x=x.copy(),
                y=y.copy(),
                x_raw=x_raw.copy(),
                y_raw=y_raw.copy(),
                s=preprocessing_output.s,
                inst_labels=preprocessing_output.inst_labels,
                prelim_options=prelim_options,
                selvars_options=selvars_options
            )
        )
        st.success("Preliminary processing completed")
        
        return prelim_output
    
    except Exception as e:
        st.error(f"Error during Prelim execution: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Visualization Functions
def visualize_binary_performance(prelim_output, selected_algorithms):
    """
    Visualize binary performance classification for selected algorithms
    """
    # Get algorithm names from session state
    algorithm_names = st.session_state.algorithm_names
    
    # Create summary statistics first
    summary_data = []
    for i, algo_idx in enumerate(selected_algorithms):
        good_count = np.sum(prelim_output.y_bin[:, algo_idx])
        total_count = prelim_output.y_bin.shape[0]
        good_percentage = (good_count / total_count) * 100
        
        summary_data.append({
            "Algorithm": algorithm_names[algo_idx],
            "Good Performance Count": good_count,
            "Good Performance (%)": f"{good_percentage:.2f}%",
            "Total Instances": total_count
        })
    
    # Display the summary
    st.subheader("Binary Performance Summary")
    st.table(pd.DataFrame(summary_data))
    
    # Bar chart instead of heatmap
    plt.figure(figsize=(10, 6))
    
    # Extract percentages for bar chart
    algorithms = [algorithm_names[i] for i in selected_algorithms]
    percentages = [np.sum(prelim_output.y_bin[:, i]) / prelim_output.y_bin.shape[0] * 100 for i in selected_algorithms]
    
    # Create bar chart
    bars = plt.bar(algorithms, percentages, color='skyblue')
    
    # Add percentage labels on top of bars
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{percentage:.1f}%", ha='center', va='bottom')
    
    plt.title("Percentage of Good Performance by Algorithm")
    plt.xlabel("Algorithms")
    plt.ylabel("Good Performance (%)")
    plt.ylim(0, 100)  # Set y-axis range from 0 to 100%
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')  # Rotate algorithm names for better readability
    plt.tight_layout()
    
    st.pyplot(plt.gcf())
    
    # Best algorithms per instance
    st.subheader("Best Algorithm Distribution")
    
    # Count best algorithm occurrences
    algorithm_counts = np.bincount(prelim_output.p.astype(int), minlength=prelim_output.y.shape[1]+1)[1:]
    
    plt.figure(figsize=(10, 6))
    algorithms = [algorithm_names[i] for i in range(len(algorithm_counts))]
    
    # Plot best algorithm distribution
    bars = plt.bar(algorithms, algorithm_counts, color='lightgreen')
    
    # Add count labels
    for bar, count in zip(bars, algorithm_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.title("Number of Instances Where Each Algorithm is Best")
    plt.xlabel("Algorithms")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')  # Rotate algorithm names for better readability
    plt.tight_layout()
    
    st.pyplot(plt.gcf())

def visualize_performance_distribution(prelim_output, selected_algorithms):
    """
    Visualize the distribution of both raw and processed performance values
    """
    # Get algorithm names from session state
    algorithm_names = st.session_state.algorithm_names
    
    plt.figure(figsize=(15, 8))
    
    # Raw performance distribution
    plt.subplot(121)
    for i in selected_algorithms:
        raw_perf = prelim_output.y_raw[:, i]
        sns.kdeplot(raw_perf, label=algorithm_names[i])
    
    plt.title("Raw Performance Distribution")
    plt.xlabel("Raw Performance Value")
    plt.ylabel("Density")
    plt.legend()
    
    # Processed performance distribution
    plt.subplot(122)
    for i in selected_algorithms:
        proc_perf = prelim_output.y[:, i]
        sns.kdeplot(proc_perf, label=algorithm_names[i])
    
    plt.title("Processed Performance Distribution")
    plt.xlabel("Processed Performance Value")
    plt.ylabel("Density")
    plt.legend()
    
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    # Display transform parameters per algorithm
    st.subheader("Algorithm Normalization Parameters")
    
    algo_data = []
    for i in selected_algorithms:
        algo_data.append({
            "Algorithm": algorithm_names[i],
            "Box-Cox λ": f"{prelim_output.lambda_y[i]:.4f}",
            "Mean (μ)": f"{prelim_output.mu_y[i]:.4f}",
            "Std Dev (σ)": f"{prelim_output.sigma_y[i]:.4f}"
        })
    
    st.table(pd.DataFrame(algo_data))
    
    # Show best performance distribution
    st.subheader("Best Performance Distribution")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(prelim_output.y_best, kde=True, color='purple')
    plt.title("Distribution of Best Performance Values")
    plt.xlabel("Best Performance Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(plt.gcf())

def visualize_feature_transformations(prelim_output, selected_features):
    """
    Visualize the effect of transformations on selected features
    """
    # Get feature names from session state
    feature_names = st.session_state.feature_names
    
    plt.figure(figsize=(15, 8))
    
    # Calculate number of rows needed for subplots
    n_features = len(selected_features)
    n_cols = 2  # Raw and transformed
    
    for i, feat_idx in enumerate(selected_features):
        # Raw feature distribution
        plt.subplot(n_features, n_cols, 2*i+1)
        raw_feat = prelim_output.x_raw[:, feat_idx]
        sns.histplot(raw_feat, kde=True)
        plt.title(f"Raw {feature_names[feat_idx]}")
        plt.xlabel("Value")
        
        # Transformed feature distribution
        plt.subplot(n_features, n_cols, 2*i+2)
        trans_feat = prelim_output.x[:, feat_idx]
        sns.histplot(trans_feat, kde=True)
        plt.title(f"Transformed {feature_names[feat_idx]}")
        plt.xlabel("Value")
    
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    # Display transformation parameters
    st.subheader("Feature Transformation Parameters")
    
    # Create transformation summary
    transform_data = []
    for feat_idx in selected_features:
        transform_data.append({
            "Feature": feature_names[feat_idx],
            "Box-Cox λ": f"{prelim_output.lambda_x[feat_idx]:.4f}",
            "Mean (μ)": f"{prelim_output.mu_x[feat_idx]:.4f}",
            "Std Dev (σ)": f"{prelim_output.sigma_x[feat_idx]:.4f}",
            "Min Value": f"{prelim_output.min_x[feat_idx]:.4f}",
            "Median": f"{prelim_output.med_val[feat_idx]:.4f}",
            "IQ Range": f"{prelim_output.iq_range[feat_idx]:.4f}",
            "Lower Bound": f"{prelim_output.lo_bound[feat_idx]:.4f}",
            "Upper Bound": f"{prelim_output.hi_bound[feat_idx]:.4f}"
        })
    
    # Display the transformation parameters
    st.table(pd.DataFrame(transform_data))
    
    # Show bounds visualization
    st.subheader("Feature Bounds Visualization")
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(selected_features))
    width = 0.2
    
    # Create labels for the x-axis using feature names
    x_labels = [feature_names[i] for i in selected_features]
    
    # Plot median values with error bars for IQ range
    plt.bar(x - width*1.5, [prelim_output.med_val[i] for i in selected_features], 
            width, label='Median', color='blue', alpha=0.7)
    
    # Plot lower bounds
    plt.bar(x - width/2, [prelim_output.lo_bound[i] for i in selected_features], 
            width, label='Lower Bound', color='red', alpha=0.7)
    
    # Plot upper bounds
    plt.bar(x + width/2, [prelim_output.hi_bound[i] for i in selected_features], 
            width, label='Upper Bound', color='green', alpha=0.7)
    
    # Plot min values
    plt.bar(x + width*1.5, [prelim_output.min_x[i] for i in selected_features], 
            width, label='Min Value', color='orange', alpha=0.7)
    
    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.title('Feature Bounds Summary')
    plt.xticks(x, x_labels, rotation=45, ha='right')  # Use feature names with rotation
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(plt.gcf())

def visualize_feature_importance(prelim_output, selected_features):
    """
    Visualize feature importance based on algorithm performance correlation
    """
    # Get feature names and algorithm names from session state
    feature_names = st.session_state.feature_names
    algorithm_names = st.session_state.algorithm_names
    
    # Calculate correlation between features and algorithm performance
    feature_importance = []
    
    for feat_idx in selected_features:
        feature_values = prelim_output.x[:, feat_idx]
        
        # Correlate with binary performance
        correlations = []
        for algo_idx in range(prelim_output.y_bin.shape[1]):
            binary_perf = prelim_output.y_bin[:, algo_idx].astype(float)
            corr = np.corrcoef(feature_values, binary_perf)[0, 1]
            correlations.append(corr)
        
        # Calculate average absolute correlation
        avg_abs_corr = np.mean(np.abs(correlations))
        
        feature_importance.append({
            "Feature": feature_names[feat_idx],
            "Correlations": correlations,
            "Average Abs Correlation": avg_abs_corr,
            "Num Good Algos": prelim_output.num_good_algos[feat_idx],
            "Beta Selected": "Yes" if prelim_output.beta[feat_idx] else "No"
        })
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x["Average Abs Correlation"], reverse=True)
    
    # Display feature importance ranking as a table
    st.subheader("Feature Importance Ranking")
    
    ranking_df = pd.DataFrame([
        {"Feature": feat["Feature"], 
         "Average Absolute Correlation": f"{feat['Average Abs Correlation']:.4f}",
         "Num Good Algorithms": f"{feat['Num Good Algos']:.2f}",
         "Beta Selected": feat["Beta Selected"]}
        for feat in feature_importance
    ])
    
    st.table(ranking_df)
    
    # Create bar chart of feature importance
    plt.figure(figsize=(10, 6))
    
    # Extract data for plotting
    features = [feat["Feature"] for feat in feature_importance]
    importance_values = [feat["Average Abs Correlation"] for feat in feature_importance]
    
    # Create horizontal bar chart
    bars = plt.barh(features, importance_values, color='teal')
    
    # Add value labels
    for bar, value in zip(bars, importance_values):
        plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{value:.3f}", va='center')
    
    plt.title("Feature Importance by Average Absolute Correlation")
    plt.xlabel("Average Absolute Correlation")
    plt.ylabel("Features")
    plt.xlim(0, max(importance_values) * 1.15)  # Give some space for the labels
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    st.pyplot(plt.gcf())
    
    # Visualize beta selection
    st.subheader("Features Selected by Beta Threshold")
    
    plt.figure(figsize=(10, 6))
    
    # Extract beta selection data
    beta_selection = [1 if prelim_output.beta[i] else 0 for i in selected_features]
    feature_names_selected = [feature_names[i] for i in selected_features]
    
    # Create bar chart with color based on selection
    colors = ['green' if b else 'red' for b in beta_selection]
    plt.bar(feature_names_selected, beta_selection, color=colors)
    
    plt.title("Feature Selection by Beta Threshold")
    plt.xlabel("Features")
    plt.ylabel("Selected (1) / Not Selected (0)")
    plt.yticks([0, 1])
    plt.xticks(rotation=45, ha='right')  # Rotate feature names
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    st.pyplot(plt.gcf())
    
    # Create correlation heatmap
    st.subheader("Feature-Algorithm Correlation Heatmap")
    
    plt.figure(figsize=(12, 8))
    
    # Create correlation matrix
    corr_matrix = np.zeros((len(selected_features), len(algorithm_names)))
    
    for i, feat_idx in enumerate(selected_features):
        feature_values = prelim_output.x[:, feat_idx]
        
        for j in range(prelim_output.y_bin.shape[1]):
            binary_perf = prelim_output.y_bin[:, j].astype(float)
            corr = np.corrcoef(feature_values, binary_perf)[0, 1]
            corr_matrix[i, j] = corr
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1,
                xticklabels=algorithm_names,
                yticklabels=[feature_names[i] for i in selected_features],
                fmt=".2f")
    
    plt.title("Feature-Algorithm Performance Correlation")
    plt.ylabel("Features")
    plt.xlabel("Algorithms")
    plt.xticks(rotation=45, ha='right')  # Rotate algorithm names
    plt.tight_layout()
    
    st.pyplot(plt.gcf())
    
    # Show number of good algorithms per feature
    st.subheader("Number of Good Algorithms per Feature")
    
    plt.figure(figsize=(12, 6))
    
    # Sort features by number of good algorithms
    sorted_indices = sorted(selected_features, 
                            key=lambda x: prelim_output.num_good_algos[x], 
                            reverse=True)
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    good_algo_counts = [prelim_output.num_good_algos[i] for i in sorted_indices]
    
    # Create bar chart
    bars = plt.bar(sorted_feature_names, good_algo_counts, color='purple')
    
    # Add count labels
    for bar, count in zip(bars, good_algo_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f"{count:.1f}", ha='center', va='bottom')
    
    plt.title("Number of Good Algorithms per Feature")
    plt.xlabel("Features")
    plt.ylabel("Number of Good Algorithms")
    plt.xticks(rotation=45, ha='right')  # Rotate feature names
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    st.pyplot(plt.gcf())

def create_prelim_output_zip(cached_output):
    """
    Create a single consolidated CSV file for transfer to the next stage.
    Preserves algorithm and feature names in the export.
    """
    # Create temporary directory
    tmp_dir = "temp_files"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create a consolidated DataFrame with all essential data
    # First, we'll create a base DataFrame with one row
    consolidated_df = pd.DataFrame({
        "total_instances": [cached_output.x.shape[0]],
        "total_features": [cached_output.x.shape[1]],
        "total_algorithms": [cached_output.y.shape[1]],
        "min_y": [float(cached_output.min_y) if cached_output.min_y is not None else 0.0]
    })
    
    # Add algorithm and feature names from session state
    if 'algorithm_names' in st.session_state:
        consolidated_df["algorithm_names"] = [json.dumps(st.session_state.algorithm_names)]
    
    if 'feature_names' in st.session_state:
        consolidated_df["feature_names"] = [json.dumps(st.session_state.feature_names)]
    
    # Helper function to add matrix to DataFrame
    def add_matrix_to_df(df, matrix, name):
        # Convert matrix to string (flattened)
        matrix_str = str(matrix.reshape(-1).tolist())
        # Convert shape to string
        shape_str = str(list(matrix.shape))
        
        # Add to DataFrame
        df[f"{name}_data"] = [matrix_str]
        df[f"{name}_shape"] = [shape_str]
        
        return df
    
    # Add all matrices
    consolidated_df = add_matrix_to_df(consolidated_df, cached_output.x, "x")
    consolidated_df = add_matrix_to_df(consolidated_df, cached_output.y, "y")
    consolidated_df = add_matrix_to_df(consolidated_df, cached_output.y_bin, "y_bin")
    consolidated_df = add_matrix_to_df(consolidated_df, cached_output.x_raw, "x_raw")
    consolidated_df = add_matrix_to_df(consolidated_df, cached_output.y_raw, "y_raw")
    
    # Add 1D arrays as strings
    if cached_output.beta is not None:
        consolidated_df["beta"] = [str(cached_output.beta.tolist())]
    if cached_output.num_good_algos is not None:
        consolidated_df["num_good_algos"] = [str(cached_output.num_good_algos.tolist())]
    if cached_output.y_best is not None:
        consolidated_df["y_best"] = [str(cached_output.y_best.tolist())]
    if cached_output.p is not None:
        consolidated_df["p"] = [str(cached_output.p.tolist())]
    
    # Add transformation parameters
    if cached_output.med_val is not None:
        consolidated_df["med_val"] = [str(cached_output.med_val.tolist())]
    if cached_output.iq_range is not None:
        consolidated_df["iq_range"] = [str(cached_output.iq_range.tolist())]
    if cached_output.hi_bound is not None:
        consolidated_df["hi_bound"] = [str(cached_output.hi_bound.tolist())]
    if cached_output.lo_bound is not None:
        consolidated_df["lo_bound"] = [str(cached_output.lo_bound.tolist())]
    if cached_output.min_x is not None:
        consolidated_df["min_x"] = [str(cached_output.min_x.tolist())]
    if cached_output.lambda_x is not None:
        consolidated_df["lambda_x"] = [str(cached_output.lambda_x.tolist())]
    if cached_output.mu_x is not None:
        consolidated_df["mu_x"] = [str(cached_output.mu_x.tolist())]
    if cached_output.sigma_x is not None:
        consolidated_df["sigma_x"] = [str(cached_output.sigma_x.tolist())]
    if cached_output.lambda_y is not None:
        consolidated_df["lambda_y"] = [str(cached_output.lambda_y.tolist())]
    if cached_output.sigma_y is not None:
        consolidated_df["sigma_y"] = [str(cached_output.sigma_y.tolist())]
    if cached_output.mu_y is not None:
        consolidated_df["mu_y"] = [str(cached_output.mu_y.tolist())]
    
    # Add instance labels and source series if available
    if hasattr(cached_output, 'instlabels') and cached_output.instlabels is not None:
        consolidated_df["instlabels"] = [str(cached_output.instlabels.tolist())]
    elif hasattr(cached_output, 'inst_labels') and cached_output.inst_labels is not None:
        consolidated_df["instlabels"] = [str(cached_output.inst_labels.tolist())]
    
    if cached_output.s is not None:
        consolidated_df["s"] = [str(cached_output.s.tolist())]
    
    # Flag for data_dense
    consolidated_df["data_dense_available"] = [cached_output.data_dense is not None]
    
    # Save to CSV
    csv_path = os.path.join(tmp_dir, "prelim_output.csv")
    consolidated_df.to_csv(csv_path, index=False)
    
    # Create zip file with just this one file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=os.path.basename(csv_path))

    # Clean up temporary file
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    # Remove the temp directory if it's empty
    try:
        os.rmdir(tmp_dir)
    except:
        pass

    # Return the zip file as bytes
    buffer.seek(0)
    return buffer.getvalue()

def show():
    """
    Main Streamlit show function for Prelim stage
    """
    st.header("Preliminary Stage: Algorithm Performance Analysis")
    
    # Load preprocessing output
    preprocessing_output = load_preprocessing_output()
    
    if preprocessing_output is None:
        st.error("Please run the Preprocessing stage first.")
        return
    
    # Extract algorithm and feature names
    algorithm_names, feature_names = extract_names_from_preprocessing(preprocessing_output)
    
    # Store names in session state for later use across all functions
    st.session_state.algorithm_names = algorithm_names
    st.session_state.feature_names = feature_names
    
    # Display detected names for verification
    with st.expander("Detected Algorithm and Feature Names", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Algorithm Names")
            for i, name in enumerate(algorithm_names):
                st.write(f"{i+1}. {name}")
        
        with col2:
            st.subheader("Feature Names")
            for i, name in enumerate(feature_names):
                st.write(f"{i+1}. {name}")
    
    # Configuration in central area instead of side column
    st.subheader("Preliminary Stage Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Performance direction
        max_perf = st.radio(
            "Performance Direction", 
            options=[True, False],
            format_func=lambda x: "Maximize Performance" if x else "Minimize Performance",
            index=0
        )
        
        # Absolute or relative performance
        abs_perf = st.radio(
            "Performance Threshold Type", 
            options=[True, False],
            format_func=lambda x: "Absolute Threshold" if x else "Relative to Best (%)",
            index=1
        )
    
    with col2:
        # Performance threshold (epsilon)
        epsilon = st.slider(
            "Performance Threshold (ε)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.01
        )
        
        # Beta threshold for algorithm selection
        beta_threshold = st.slider(
            "Beta Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.01
        )
    
    with col3:
        # Data processing options
        bound = st.checkbox("Remove Outliers", value=True)
        norm = st.checkbox("Normalize Data", value=True)
        
        # Advanced options in a more compact form
        small_scale_flag = st.checkbox("Use Small Scale Experiment", value=False)
        density_flag = st.checkbox("Use Density-based Filtering", value=False)
    
    # Run button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Run Preliminary Analysis", type="primary", use_container_width=True):
            with st.spinner("Running preliminary analysis..."):
                # Create default values for advanced options
                small_scale = 0.5
                min_distance = 0.5
                selvars_type = "manual"
                
                # Create options objects
                prelim_options = PrelimOptions(
                    max_perf=max_perf,
                    abs_perf=abs_perf,
                    epsilon=epsilon,
                    beta_threshold=beta_threshold,
                    bound=bound,
                    norm=norm
                )
                
                selvars_options = SelvarsOptions(
                    feats=None,
                    algos=None,
                    small_scale_flag=small_scale_flag,
                    small_scale=small_scale,
                    file_idx_flag=False,
                    file_idx=None,
                    selvars_type=selvars_type,
                    min_distance=min_distance,
                    density_flag=density_flag
                )
                
                # Run preliminary stage
                prelim_output = run_prelim(
                    preprocessing_output, 
                    prelim_options,
                    selvars_options
                )
                
                # Store the result in session state for visualization
                if prelim_output is not None:
                    st.session_state.prelim_output = prelim_output
                    st.session_state.prelim_options = prelim_options
                    st.session_state.selvars_options = selvars_options
                    
                    # Save the output directly for the next stage
                    # Use our custom saving function instead of save_to_cache
                    if manually_save_output(prelim_output):
                        st.success("Preliminary stage completed. Data saved for the next stage.")
                    else:
                        st.warning("Could not save data for next stage. The analysis results are still available for viewing.")
    
    # Visualization section
    if 'prelim_output' in st.session_state:
        prelim_output = st.session_state.prelim_output
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Instances", prelim_output.x.shape[0])
        with col2:
            st.metric("Total Features", prelim_output.x.shape[1])
        with col3:
            st.metric("Total Algorithms", prelim_output.y.shape[1])
        
        # Display parameter choices
        st.subheader("Analysis Parameters")
        
        params_dict = {
            "Performance Direction": "Maximize" if st.session_state.prelim_options.max_perf else "Minimize",
            "Threshold Type": "Absolute" if st.session_state.prelim_options.abs_perf else "Relative (%)",
            "Epsilon Threshold": f"{st.session_state.prelim_options.epsilon:.2f}",
            "Beta Threshold": f"{st.session_state.prelim_options.beta_threshold:.2f}",
            "Outlier Removal": "Enabled" if st.session_state.prelim_options.bound else "Disabled",
            "Normalization": "Enabled" if st.session_state.prelim_options.norm else "Disabled"
        }
        
        st.table(pd.DataFrame([params_dict]).T.rename(columns={0: "Value"}))
        
        # Tabs for different visualization aspects
        tab1, tab2, tab3, tab4 = st.tabs([
            "Binary Performance", 
            "Performance Distribution", 
            "Feature Transformations", 
            "Feature Importance"
        ])
        
        with tab1:
            # Algorithm selection with names
            num_algorithms = prelim_output.y.shape[1]
            algorithm_options = list(range(num_algorithms))
            
            selected_algorithms = st.multiselect(
                "Select Algorithms to Visualize", 
                options=algorithm_options, 
                default=algorithm_options[:min(5, num_algorithms)],
                format_func=lambda x: st.session_state.algorithm_names[x]
            )
            
            if selected_algorithms:
                visualize_binary_performance(prelim_output, selected_algorithms)
            else:
                st.info("Please select at least one algorithm to visualize")
        
        with tab2:
            # Algorithm selection with names
            num_algorithms = prelim_output.y.shape[1]
            algorithm_options = list(range(num_algorithms))
            
            selected_algorithms = st.multiselect(
                "Select Algorithms to Visualize", 
                options=algorithm_options, 
                default=algorithm_options[:min(3, num_algorithms)],
                format_func=lambda x: st.session_state.algorithm_names[x],
                key="perf_dist_algos"
            )
            
            if selected_algorithms:
                visualize_performance_distribution(prelim_output, selected_algorithms)
            else:
                st.info("Please select at least one algorithm to visualize")
        
        with tab3:
            # Feature selection with names
            num_features = prelim_output.x.shape[1]
            feature_options = list(range(num_features))
            
            selected_features = st.multiselect(
                "Select Features to Visualize", 
                options=feature_options, 
                default=feature_options[:min(3, num_features)],
                format_func=lambda x: st.session_state.feature_names[x]
            )
            
            if selected_features:
                visualize_feature_transformations(prelim_output, selected_features)
            else:
                st.info("Please select at least one feature to visualize")
        
        with tab4:
            # Feature selection with names
            num_features = prelim_output.x.shape[1]
            feature_options = list(range(num_features))
            
            selected_features = st.multiselect(
                "Select Features to Visualize", 
                options=feature_options, 
                default=feature_options[:min(5, num_features)],
                format_func=lambda x: st.session_state.feature_names[x],
                key="feat_imp_features"
            )
            
            if selected_features:
                visualize_feature_importance(prelim_output, selected_features)
            else:
                st.info("Please select at least one feature to visualize")
        
        # Download Section
        st.subheader("Download Results")
        
        # Create download button with single CSV file
        zip_data = create_prelim_output_zip(prelim_output)
        
        st.download_button(
            label="Download Prelim Output (ZIP)",
            data=zip_data,
            file_name="prelim_output.zip",
            mime="application/zip",
            help="Download comprehensive Prelim stage output"
        )
        
        # Success message for next stage
        st.success("Preliminary stage completed. You can proceed to the next stage (SIFTED).")
    else:
        # Instructions
        st.info("Configure parameters and click 'Run Preliminary Analysis' to begin.")

# Run the main function
if __name__ == "__main__":
    show()