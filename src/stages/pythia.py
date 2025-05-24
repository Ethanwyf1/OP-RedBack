import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from utils.cache_utils import cache_exists, delete_cache, load_from_cache, save_to_cache
from utils.download_utils import create_stage_output_zip
from utils.run_pythia import run_pythia
from instancespace.data.options import PythiaOptions, ParallelOptions
from collections import Counter

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
        st.caption("⚖️ Enable polynomial kernel for SVM. Recommended for larger datasets or complex boundaries.")

        st.checkbox("Use Cost-Sensitive Weights", value=False, disabled=True)
        st.caption("🖐️ *This feature is under development and currently not supported in model training.*")

        use_bayes_opt = st.checkbox(
            "Use Bayesian Optimization (default: Grid Search)", 
            value=False, 
            help="If checked, Bayesian Optimization will be used instead of Grid Search."
        )
        if use_bayes_opt:
            st.caption("🧠 Bayesian Optimization: Efficient for large or continuous hyperparameter spaces. "
                       "Faster convergence but may miss global optima in some cases.")
        else:
            st.caption("🔍 Grid Search (default): Exhaustive search over a fixed hyperparameter grid. "
                       "More reliable for small search spaces but slower.")

        use_parallel = st.checkbox("Enable Parallel Training", value=False)
        st.caption("⚡ Enable parallel SVM training. Useful for large datasets with many algorithms.")

        n_cores = st.number_input("Number of Cores (for parallelism)", min_value=1, max_value=16, value=1)
        st.caption("💻 Number of CPU cores to use for parallel processing. Only applies when parallel training is enabled.")

    # Construct options objects
    pythia_opts = PythiaOptions(
        cv_folds=cv_folds,
        is_poly_krnl=use_poly_kernel,
        use_weights=False,
        use_grid_search=not use_bayes_opt,
        params=None
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

    algo_labels = output.pythia_summary["Algorithms"][:len(output.accuracy)]

    # --- Step 6: Visualization Controls ---
    st.subheader("📽️ Predicted Best Algorithm in 2D Instance Space")
    pilot_output = load_from_cache("pilot_output.pkl")
    Z = pilot_output.z

    selection_mode = st.radio(
        "🎯 Which algorithm selection to visualize?",
        options=["Primary Selection (selection0)", "Fallback Selection (selection1)"],
        index=0,
        horizontal=True
    )
    algo_indices = output.selection0 if selection_mode.startswith("Primary") else output.selection1

    visible_algos = st.multiselect(
        "🔍 Select algorithms to highlight (others will be dimmed)",
        options=algo_labels,
        default=algo_labels
    )

    filter_mode = st.radio(
        "🔍 Filter by Prediction Outcome",
        options=["Show All", "Only Predicted Good (1)", "Only Predicted Bad (0)"],
        index=0,
        horizontal=True
    )

    y_hat = output.y_hat
    assert len(algo_indices) == y_hat.shape[0]
    predictions = np.array([
        y_hat[i, algo_indices[i]] if 0 <= algo_indices[i] < y_hat.shape[1] else -1
        for i in range(len(algo_indices))
    ])

    num_total = np.sum((predictions == 0) | (predictions == 1))
    num_good = np.sum(predictions == 1)
    num_bad = np.sum(predictions == 0)
    num_invalid = np.sum(predictions == -1)

    # Create a one-row dataframe with prediction summary (clean version without emojis)
    summary_df = pd.DataFrame([{
        "Good (1)": str(num_good),
        "Bad (0)": str(num_bad),
        "Invalid": str(num_invalid),
        "Total Valid": f"{num_total} / {len(predictions)}"
    }])

    st.subheader("Prediction Summary")
    st.dataframe(summary_df, use_container_width=True)
    
    algo_counts = Counter(algo_indices)
    label_map = output.pythia_summary["Algorithms"]
    st.markdown("**🧴 Algorithm Selection Frequency:**")
    for idx, count in algo_counts.items():
        label = label_map[idx] if idx < len(label_map) else f"[invalid idx {idx}]"
        st.markdown(f"- `{label}`: {count} instances selected")

    df_viz = pd.DataFrame({
        "Z1": Z[:, 0],
        "Z2": Z[:, 1],
        "Selected Algorithm": [algo_labels[i] for i in algo_indices],
        "Prediction": predictions
    })

    if filter_mode == "Only Predicted Good (1)":
        df_viz = df_viz[df_viz["Prediction"] == 1]
    elif filter_mode == "Only Predicted Bad (0)":
        df_viz = df_viz[df_viz["Prediction"] == 0]

    df_viz["Color"] = [
        algo if algo in visible_algos else "Dimmed"
        for algo in df_viz["Selected Algorithm"]
    ]

    color_map = {
        "Dimmed": "lightgrey",
        **{label: px.colors.qualitative.Bold[i % 10] for i, label in enumerate(algo_labels)}
    }

    fig = px.scatter(
        df_viz,
        x="Z1",
        y="Z2",
        color="Color",
        title="Predicted Best Algorithm per Instance (Filtered by y_hat)",
        color_discrete_map=color_map,
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
    autosize=False,
    width=800,
    height=800
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Step 7: Summary Table ---
    st.subheader("📋 Summary of All Algorithm Performances")
    summary_df = output.pythia_summary.copy()
    summary_df = summary_df.replace("", np.nan).infer_objects(copy=False)
    st.dataframe(summary_df, use_container_width=True)

    # --- Step 8: Download ---
    st.subheader("📅 Download Cached PYTHIA Output")
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

    # --- Step 9: Cache Management ---
    st.subheader("🗑️ Cache Management")
    if st.button("❌ Delete PYTHIA Cache"):
        success = delete_cache("pythia_output.pkl")
        if success:
            st.success("🗑️ PYTHIA cache deleted.")
        else:
            st.warning("⚠️ No PYTHIA cache file found to delete.")