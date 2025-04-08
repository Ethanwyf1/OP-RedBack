import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from instancespace.data.options import SelvarsOptions
from utils.run_preprocessing import run_preprocessing


def show():
    st.header("🔍 Preprocessing Stage")

    # --- User Inputs ---
    feats_input = st.text_input("Enter selected features (comma separated)")
    algos_input = st.text_input("Enter selected algorithms (comma separated)")
    normalize_flag = st.checkbox("Apply normalization")

    # Dropdown menu to choose normalization method (only enabled if normalization is checked)
    normalization_method = None
    if normalize_flag:
        normalization_method = st.selectbox(
            "Select normalization method",
            options=["StandardScaler", "MinMaxScaler", "RobustScaler"]
        )

    run_button = st.button("Run Preprocessing")

    if run_button:
        feats = [f.strip() for f in feats_input.split(",") if f.strip()] or None
        algos = [a.strip() for a in algos_input.split(",") if a.strip()] or None

        # Prepare options for ISA
        selvars = SelvarsOptions(
            feats=feats,
            algos=algos,
            small_scale_flag=False,
            small_scale=0.1,
            file_idx_flag=False,
            file_idx=None,
            selvars_type="manual",
            min_distance=0.0,
            density_flag=False
        )

        # --- Run Preprocessing ---
        output = run_preprocessing("data/metadata.csv", feats=feats, algos=algos)

        # --- Summary Table ---
        summary_data = {
            "Metric": [
                "Total Instances",
                "Total Features",
                "Total Algorithms",
                "Selected Features",
                "Selected Algorithms",
                "Missing Values (X)",
                "Final Feature Shape",
                "Final Algorithm Shape",
                "Normalization Applied",
                "Normalization Method"
            ],
            "Value": [
                output.x.shape[0],
                output.x_raw.shape[1],
                output.y_raw.shape[1],
                len(output.feat_labels),
                len(output.algo_labels),
                int(np.isnan(output.x).sum()),
                str(output.x.shape),
                str(output.y.shape),
                "Yes" if normalize_flag else "No",
                normalization_method if normalize_flag else "N/A"
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        st.subheader("📊 Data Summary")
        st.dataframe(df_summary, use_container_width=True)

        # --- Boxplot Visualization ---
        st.subheader("📈 Feature Distribution Comparison")

        feature_df_original = pd.DataFrame(output.x, columns=output.feat_labels)

        # Normalize features for visualization only (does NOT affect pipeline logic)
        if normalize_flag:
            if normalization_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif normalization_method == "RobustScaler":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            x_normalized = scaler.fit_transform(output.x)
            feature_df_normalized = pd.DataFrame(x_normalized, columns=output.feat_labels)

            # Display side-by-side boxplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

            sns.boxplot(data=feature_df_original, ax=axes[0])
            axes[0].set_title("Before Normalization")
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

            sns.boxplot(data=feature_df_normalized, ax=axes[1])
            axes[1].set_title("After Normalization")
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

            st.pyplot(fig)

            # Warning note
            st.info("⚠️ Note: Normalization is applied only for visualization and does not modify the pipeline output.")
        else:
            # Show only original boxplot
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=feature_df_original, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)

        st.success("Preprocessing completed and visualized successfully.")
