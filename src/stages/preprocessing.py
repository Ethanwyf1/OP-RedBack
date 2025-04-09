import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from instancespace.data.options import SelvarsOptions
from utils.run_preprocessing import run_preprocessing


def show():
    st.header("🔍 Preprocessing Stage")

    # Step 1: Load initial data (for selection options)
    default_output = run_preprocessing("data/metadata.csv")

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
        # Build SelvarsOptions
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

        output = run_preprocessing("data/metadata.csv", feats=selected_feats, algos=selected_algos)

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
                "Final Algorithm Shape"
            ],
            "Value": [
                output.x.shape[0],
                output.x_raw.shape[1],
                output.y_raw.shape[1],
                len(output.feat_labels),
                len(output.algo_labels),
                int(np.isnan(output.x).sum()),
                str(output.x.shape),
                str(output.y.shape)
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        st.subheader("📊 Data Summary")
        st.dataframe(df_summary, use_container_width=True)

        # --- Boxplot Visualization ---
        st.subheader("📈 Feature Distributions")
        feature_df = pd.DataFrame(output.x, columns=output.feat_labels)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=feature_df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

        st.success("✅ Preprocessing completed and visualized.")
