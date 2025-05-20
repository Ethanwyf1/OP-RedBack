import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from utils.cache_utils import cache_exists, delete_cache, load_from_cache, save_to_cache
from utils.download_utils import create_stage_output_zip
from utils.run_pythia import run_pythia

def show():
    st.header("🤖 PYTHIA Stage – Automated Algorithm Selection")

    # --- Step 1: Check Dependencies ---
    required_cache = ["pilot_output.pkl", "prelim_output.pkl", "sifted_output.pkl", "preprocessing_output.pkl"]

    missing = [f for f in required_cache if not cache_exists(f)]
    if missing:
        st.error(f"🚫 Required inputs not found: {', '.join(missing)}. Please run previous stages first.")
        return

    # --- Step 2: Run PYTHIA Button ---
    if st.button("Run PYTHIA", key="run_pythia_btn"):
        output = run_pythia()
        st.session_state["pythia_output"] = output
        save_to_cache(output, "pythia_output.pkl")
        st.toast("✅ PYTHIA stage completed successfully!", icon="🤖")

    # --- Step 3: Load Output ---
    output = None
    if "pythia_output" in st.session_state:
        output = st.session_state["pythia_output"]
    elif cache_exists("pythia_output.pkl"):
        output = load_from_cache("pythia_output.pkl")

    if output is None:
        st.warning("⚠️ PYTHIA output not available. Please click **Run PYTHIA**.")
        return

    st.success("✅ PYTHIA Output Loaded")

    # --- Step 4: Metrics Summary Table ---
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

    # --- Step 5: Prediction Visualization ---
    st.subheader("🗺️ Predicted Best Algorithm in 2D Instance Space")

    pilot_output = load_from_cache("pilot_output.pkl")
    Z = pilot_output.z

    df_viz = pd.DataFrame({
        "Z1": Z[:, 0],
        "Z2": Z[:, 1],
        "Selected Algorithm": [output.pythia_summary["Algorithms"][i] for i in output.selection0],
    })

    fig = px.scatter(
        df_viz,
        x="Z1",
        y="Z2",
        color="Selected Algorithm",
        title="Predicted Best Algorithm per Instance (Based on SVM Classifier)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Step 6: Summary Table ---
    st.subheader("📋 Summary of All Algorithm Performances")
    summary_df = output.pythia_summary.copy()
    summary_df = summary_df.replace("", np.nan).infer_objects(copy=False)
    st.dataframe(summary_df, use_container_width=True)

    # --- Step 7: Download Results ---
    st.subheader("📥 Download Cached PYTHIA Output")

    if cache_exists("pythia_output.pkl"):
        df_features = pd.DataFrame(output.w)
        df_probs = pd.DataFrame(
            output.pr0_hat, 
            columns=output.pythia_summary["Algorithms"][:len(output.accuracy)]
        )

        # ✅ Load instance labels from preprocessing_output.pkl
        preprocessing_output = load_from_cache("preprocessing_output.pkl")
        instance_labels = getattr(preprocessing_output, "inst_labels", None)

        zip_data = create_stage_output_zip(
            x=df_features,
            y=df_probs,
            instance_labels=instance_labels,  # ✅ Fixed here
            source_labels=None,
            metadata_description="Cached output from PYTHIA stage (including predictions and classifier results).",
        )

        st.download_button(
            label="⬇️ Download PYTHIA Output (ZIP)",
            data=zip_data,
            file_name="pythia_output.zip",
            mime="application/zip"
        )
    else:
        st.warning("⚠️ No cached PYTHIA output found.")

    # --- Step 8: Delete Cache ---
    st.subheader("🗑️ Cache Management")
    if st.button("❌ Delete PYTHIA Cache"):
        success = delete_cache("pythia_output.pkl")
        if success:
            st.success("🗑️ PYTHIA cache deleted.")
        else:
            st.warning("⚠️ No PYTHIA cache file found to delete.")
