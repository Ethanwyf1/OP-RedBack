# stages/pythia.py

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from utils.cache_utils import cache_exists, delete_cache, load_from_cache, save_to_cache
from utils.download_utils import create_stage_output_zip
from utils.run_pythia import run_pythia
from instancespace.data.options import PythiaOptions, ParallelOptions


def show():
    st.header("🤖 PYTHIA Stage – Automated Algorithm Selection")

    # --- Step 1: Check Dependencies ---
    required_cache = ["pilot_output.pkl", "prelim_output.pkl", "sifted_output.pkl", "preprocessing_output.pkl"]
    missing = [f for f in required_cache if not cache_exists(f)]
    if missing:
        st.error(f"🚫 Required inputs not found: {', '.join(missing)}. Please run previous stages first.")
        return

    # --- Step 2: Allow user to configure PYTHIA Options ---
    with st.expander("⚙️ PYTHIA Configuration", expanded=False):
        cv_folds = st.slider("Number of CV folds", min_value=2, max_value=10, value=5)
        st.caption("🔁 Controls how many folds are used in cross-validation. More folds improve accuracy estimation but increase runtime.")

        use_poly_kernel = st.checkbox("Use Polynomial Kernel", value=False)
        st.caption("📐 Enable polynomial kernel for SVM. Recommended for larger datasets or complex boundaries.")

        use_weights = st.checkbox("Use Cost-Sensitive Weights", value=True)
        st.caption("⚖️ Give more weight to hard-to-classify instances. Useful for imbalanced datasets.")

        use_grid_search = st.checkbox("Use Grid Search (instead of Bayesian Optimization)", value=True)
        st.caption("🔍 Use grid search for hyperparameter tuning. More exhaustive but slower than Bayesian search.")

        use_parallel = st.checkbox("Enable Parallel Training", value=False)
        st.caption("⚡ Enable parallel SVM training. Useful for large datasets with many algorithms.")

        n_cores = st.number_input("Number of Cores (for parallelism)", min_value=1, max_value=16, value=1)
        st.caption("💻 Number of CPU cores to use for parallel processing. Only applies when parallel training is enabled.")


    # Construct options objects
    pythia_opts = PythiaOptions(
        cv_folds=cv_folds,
        is_poly_krnl=use_poly_kernel,
        use_weights=use_weights,
        use_grid_search=use_grid_search,
        params=None  # Optional: support for manually passed hyperparameters
    )
    parallel_opts = ParallelOptions(
        flag=use_parallel,
        n_cores=n_cores
    )

    # --- Step 3: Run PYTHIA ---
    if st.button("🚀 Run PYTHIA", key="run_pythia_btn"):
        output = run_pythia(
            pythia_options=pythia_opts,
            parallel_options=parallel_opts
        )
        st.session_state["pythia_output"] = output
        save_to_cache(output, "pythia_output.pkl")
        st.toast("✅ PYTHIA stage completed successfully!", icon="🤖")

    # --- Step 4: Load Output ---
    output = st.session_state.get("pythia_output") if "pythia_output" in st.session_state else \
        (load_from_cache("pythia_output.pkl") if cache_exists("pythia_output.pkl") else None)

    if output is None:
        st.warning("⚠️ PYTHIA output not available. Please click **Run PYTHIA**.")
        return

    st.success("✅ PYTHIA Output Loaded")

    # --- Step 5: Metrics Summary Table ---
    st.subheader("📊 Classification Performance by Algorithm")
    algo_labels = output.pythia_summary["Algorithms"][:len(output.accuracy)]

    df_metrics = pd.DataFrame({
        "Algorithm": algo_labels,
        "Accuracy (%)": np.round(100 * np.array(output.accuracy), 2),
        "Precision (%)": np.round(100 * np.array(output.precision), 2),
        "Recall (%)": np.round(100 * np.array(output.recall), 2),
        "Box Constraint (C)": output.box_consnt,
        "Kernel Scale (γ)": output.k_scale
    })
    st.dataframe(df_metrics, use_container_width=True)

    # --- Step 6: Prediction Visualization ---
    st.subheader("🗺️ Predicted Best Algorithm in 2D Instance Space")

    pilot_output = load_from_cache("pilot_output.pkl")
    Z = pilot_output.z
    algo_indices = output.selection0

    if Z.shape[0] != len(algo_indices):
        st.error(f"⚠️ Shape mismatch: Z has {Z.shape[0]} instances but selection0 has {len(algo_indices)}")
        return

    df_viz = pd.DataFrame({
        "Z1": Z[:, 0],
        "Z2": Z[:, 1],
        "Selected Algorithm": [output.pythia_summary["Algorithms"][i] for i in algo_indices],
    })

    fig = px.scatter(
        df_viz,
        x="Z1",
        y="Z2",
        color="Selected Algorithm",
        title="Predicted Best Algorithm per Instance (Based on SVM Classifier)",
        opacity=0.6, 
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)

    # --- Step 7: Summary Table ---
    st.subheader("📋 Summary of All Algorithm Performances")
    summary_df = output.pythia_summary.copy()
    summary_df = summary_df.replace("", np.nan).infer_objects(copy=False)
    st.dataframe(summary_df, use_container_width=True)

    # --- Step 8: Download Results ---
    st.subheader("📥 Download Cached PYTHIA Output")
    if cache_exists("pythia_output.pkl"):
        df_features = pd.DataFrame(output.w)
        df_probs = pd.DataFrame(
            output.pr0_hat,
            columns=output.pythia_summary["Algorithms"][:len(output.accuracy)]
        )
        preprocessing_output = load_from_cache("preprocessing_output.pkl")
        instance_labels = getattr(preprocessing_output, "inst_labels", None)

        zip_data = create_stage_output_zip(
            x=df_features,
            y=df_probs,
            instance_labels=instance_labels,
            source_labels=None,
            metadata_description="Cached output from PYTHIA stage (including predictions and classifier results)."
        )

        st.download_button(
            label="⬇️ Download PYTHIA Output (ZIP)",
            data=zip_data,
            file_name="pythia_output.zip",
            mime="application/zip"
        )
    else:
        st.warning("⚠️ No cached PYTHIA output found.")

    # --- Step 9: Delete Cache ---
    st.subheader("🗑️ Cache Management")
    if st.button("❌ Delete PYTHIA Cache"):
        success = delete_cache("pythia_output.pkl")
        if success:
            st.success("🗑️ PYTHIA cache deleted.")
        else:
            st.warning("⚠️ No PYTHIA cache file found to delete.")
