import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from instancespace.data.options import SelvarsOptions
from utils.run_preprocessing import run_preprocessing


def show():
    st.header("🔍 Preprocessing Stage")

    uploaded_file = st.session_state.get("uploaded_file")
    if uploaded_file is None:
        st.error("🚫 Please upload the metadata CSV file on the Homepage page before using the Preprocessing function.")
        return
    else:
        metadata_input = uploaded_file

    # Step 1: Load initial data (for selection options)
    default_output = run_preprocessing(metadata_input)

    st.markdown("This stage performs data cleaning and filtering. Use the options below to select features and algorithms.")

    # Step 2: Feature Selection via checkbox
    st.subheader("🧬 Feature Selection")
    selected_feats = []
    with st.expander("Select features", expanded=False):
        for feat in default_output.feat_labels:
            if st.checkbox(f"{feat}", value=True, key=f"feat_{feat}"):
                selected_feats.append(feat)

    # Step 3: Algorithm Selection via checkbox
    st.subheader("⚙️ Algorithm Selection")
    selected_algos = []
    with st.expander("Select algorithms", expanded=False):
        for algo in default_output.algo_labels:
            if st.checkbox(f"{algo}", value=True, key=f"algo_{algo}"):
                selected_algos.append(algo)

    # Run button
    run_button = st.button("Run Preprocessing", key="run_preprocessing_btn")

    if run_button:
        selvars = SelvarsOptions(
            feats=selected_feats if selected_feats else None,
            algos=selected_algos if selected_algos else None,
            small_scale_flag=False,
            small_scale=0.1,
            file_idx_flag=False,
            file_idx=None,
            selvars_type="manual",
            min_distance=0.0,
            density_flag=False
        )

        st.session_state["output"] = run_preprocessing("data/metadata.csv", feats=selected_feats, algos=selected_algos)
        st.session_state["ran_preprocessing"] = True

    if st.session_state.get("ran_preprocessing", False):

        output = st.session_state["output"]

        # --- Summary ---
        summary_data = {
            "Metric": [
                "Number of Instances",
                "Number of Features (Raw)",
                "Number of Algorithms (Raw)",
                "Selected Features",
                "Selected Algorithms",
                "Missing Values (Features)",
                "Processed Feature Matrix",
                "Processed Algorithm Matrix"
            ],
            "Value": [
                output.x.shape[0],
                output.x_raw.shape[1],
                output.y_raw.shape[1],
                len(output.feat_labels),
                len(output.algo_labels),
                int(np.isnan(output.x).sum()),
                f"{output.x.shape[0]} rows × {output.x.shape[1]} columns",
                f"{output.y.shape[0]} rows × {output.y.shape[1]} columns"
            ],
            "Description": [
                "Total number of problem instances uploaded",
                "Total number of original input features before selection",
                "Total number of algorithms in the dataset",
                "Features selected for further analysis (SIFTED stage)",
                "Algorithms selected for performance comparison (PILOT, PYTHIA)",
                "Total missing values in feature columns (affects preprocessing)",
                "Dimensions of cleaned and transformed feature data",
                "Dimensions of filtered algorithm performance data"
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        st.subheader("📊 Data Summary")
        st.dataframe(df_summary, use_container_width=True)

        # --- Feature Detail ---
        with st.expander("📋 Detailed Feature Summary (Click to Expand)", expanded=False):
            feature_stats = []
            feature_df = pd.DataFrame(output.x, columns=output.feat_labels)
            for feat in feature_df.columns:
                col = feature_df[feat]
                feature_stats.append({
                    "Feature Name": feat,
                    "Valid Count": int(((col != 0) & col.notna()).sum()),
                    "Missing Values": int(col.isna().sum()),
                    "Min": round(float(col.min()), 4),
                    "Max": round(float(col.max()), 4),
                    "Mean": round(float(col.mean()), 4),
                    "Std": round(float(col.std()), 4)
                })
            df_feature_stats = pd.DataFrame(feature_stats)
            st.dataframe(df_feature_stats, use_container_width=True)

        # --- Boxplot Visualization ---
        st.subheader("📈 Feature Distributions")

        use_log_scale = st.checkbox("Use logarithmic scale (recommended for large value ranges)", value=False)
        show_outliers = st.checkbox("Show outliers (may clutter view)", value=True)

        fig_height = max(5, len(feature_df.columns) * 0.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        sns.boxplot(
            data=feature_df,
            ax=ax,
            showfliers=show_outliers
        )

        if use_log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Feature Value (log scale)")
        else:
            ax.set_ylabel("Feature Value")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Distribution of Selected Features")
        st.pyplot(fig)

        st.caption("ℹ️ Use log scale to better visualize features with extreme values. Outliers may distort box shape.")
        st.success("✅ Preprocessing completed and visualized.")