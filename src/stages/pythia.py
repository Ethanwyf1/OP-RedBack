import os
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from instancespace.data.options import ParallelOptions, PythiaOptions
from utils.cache_utils import cache_exists, delete_cache, load_from_cache, save_to_cache
from utils.download_utils import create_stage_output_zip
from utils.run_pythia import run_pythia

# ─────────────────────────────────────────────
# Helper colour maps
# ─────────────────────────────────────────────
DISCRETE_LABEL_COLORS = {"Good": "#2c83a0", "Bad": "#d6278a"}


def _algo_colors(labels: list[str]) -> dict[str, str]:
    palette = px.colors.qualitative.Bold
    return {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}


# ─────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────
def make_portfolio_table(
    selection0: np.ndarray, selection1: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """Return table of counts per algorithm for selection0 / selection1."""
    c0, c1 = Counter(selection0), Counter(selection1)
    return pd.DataFrame(
        {
            "Algorithm": labels,
            "Primary (selection0)": [c0.get(i, 0) for i in range(len(labels))],
            "Fallback (selection1)": [c1.get(i, 0) for i in range(len(labels))],
        }
    )


def make_individual_table(label_vector: np.ndarray, algo_name: str) -> pd.DataFrame:
    """Return good/bad counts for a single algorithm."""
    good = int(np.sum(label_vector == 1))
    bad = int(np.sum(label_vector == 0))
    return pd.DataFrame(
        [
            {
                "Algorithm": algo_name,
                "Good (1)": good,
                "Bad (0)": bad,
                "Total": good + bad,
            }
        ]
    )


# ─────────────────────────────────────────────
# Streamlit page entry-point
# ─────────────────────────────────────────────
def show() -> None:
    st.header("🤖 PYTHIA Stage – Automated Algorithm Selection")

    # --- dependency check ------------------------------------------------------
    required_cache = [
        "pilot_output.pkl",
        "prelim_output.pkl",
        "sifted_output.pkl",
        "preprocessing_output.pkl",
    ]
    missing = [f for f in required_cache if not cache_exists(f)]
    if missing:
        st.error(
            f"🚫 Required inputs not found: {', '.join(missing)}. "
            "Please run previous stages first."
        )
        return

 # --- configuration expander ---------------------------------------
    with st.expander("⚙️ PYTHIA Configuration", expanded=False):
        cv_folds = st.slider("Number of CV folds", 2, 10, 5)
        st.caption("🔁 Controls how many folds are used in cross-validation.")

        use_poly_kernel = st.checkbox("Use Polynomial Kernel", value=False)
        st.caption("⚖️ Enable polynomial kernel for SVM.")

        st.checkbox("Use Cost-Sensitive Weights", value=False, disabled=True)
        st.caption("🖐️ Cost-sensitive weights not supported yet.")

        use_bayes_opt = st.checkbox(
            "Use Bayesian Optimization (default: Grid Search)", value=False
        )
        st.caption("🧠 Bayesian Optimization is fast for wide search spaces."
                   if use_bayes_opt else "🔍 Grid Search exhaustively explores a fixed grid.")

        use_parallel = st.checkbox("Enable Parallel Training", value=False)
        n_cores = st.number_input("Number of Cores", 1, mp.cpu_count(), 1)

    pythia_opts = PythiaOptions(
        cv_folds=cv_folds,
        is_poly_krnl=use_poly_kernel,
        use_weights=False,
        use_grid_search=not use_bayes_opt,
        params=None,
    )
    parallel_opts = ParallelOptions(flag=use_parallel, n_cores=int(n_cores))

    # --- run PYTHIA ------------------------------------------------------------
    if st.button("🚀 Run PYTHIA", key="run_pythia_btn") or not cache_exists(
        "pythia_output.pkl"
    ):
        with st.spinner("Running PYTHIA – this may take a while …"):
            output = run_pythia(
                pythia_options=pythia_opts, parallel_options=parallel_opts
            )
        save_to_cache(output, "pythia_output.pkl")
        st.session_state["pythia_output"] = output
        st.toast("✅ PYTHIA stage completed successfully!", icon="🤖")

    # --- load outputs ----------------------------------------------------------
    output = st.session_state.get("pythia_output") or load_from_cache(
        "pythia_output.pkl"
    )
    if output is None:
        st.warning("⚠️ PYTHIA output not available. Please run the stage.")
        return
    st.success("✅ PYTHIA Output Loaded")

    preprocessing_output = load_from_cache("preprocessing_output.pkl")
    pilot_out = load_from_cache("pilot_output.pkl")
    prelim_out = load_from_cache("prelim_output.pkl")

    algo_labels: list[str] = list(getattr(preprocessing_output, "algo_labels", []))
    if not algo_labels:
        algo_labels = output.pythia_summary["Algorithms"][: len(output.accuracy)].tolist()

    algo_color_map = _algo_colors(algo_labels)
    Z = pilot_out.z

    # --- visualisation controls ------------------------------------------------
    st.subheader("📽️ 2-D Instance Space Visualisation")

    if vis_mode := st.radio("Analysis Mode", ["Individual Algorithm", "Portfolio"]):
        if vis_mode == "Individual Algorithm":
            data_src = st.radio(
                "Data Source",
                ["Training (ground-truth)", "Prediction (CV prediction)", "Prediction (full-fit prediction)"],
                index=0,
            )
        else:  # Portfolio
            data_src = st.radio(
                "Data Source",
                ["Training (y_best)", "Prediction"],   # ← only one training flavour
                index=0,
            )

    # ============================== INDIVIDUAL =================================
    if vis_mode.startswith("Individual"):
        algo = st.selectbox("Choose algorithm", algo_labels)
        k = algo_labels.index(algo)

        if data_src.startswith("Training (ground"):
            labels_bin = prelim_out.y_bin[:, k].astype(int)  # ground-truth labels
            title_suffix = "TRAINING (y_bin: ground-truth)"
        elif data_src.startswith("Prediction (CV"):
            labels_bin = output.y_sub[:, k].astype(int)  # CV prediction
            title_suffix = "PREDICTION (y_sub: CV prediction)"
        else:  # Prediction
            labels_bin = output.y_hat[:, k].astype(int)  # full-fit prediction
            title_suffix = "PREDICTION (y_hat: full-fit)"

        labels_str = np.where(labels_bin == 1, "Good", "Bad")
        filt = st.radio("Filter", ["All", "Only Good", "Only Bad"], horizontal=True)
        mask = (
            (labels_str == "Good")
            if filt == "Only Good"
            else (labels_str == "Bad")
            if filt == "Only Bad"
            else np.ones_like(labels_str, dtype=bool)
        )

        df_viz = pd.DataFrame(
            {"Z1": Z[mask, 0], "Z2": Z[mask, 1], "Label": labels_str[mask]}
        )
        fig = px.scatter(
            df_viz,
            x="Z1",
            y="Z2",
            color="Label",
            color_discrete_map=DISCRETE_LABEL_COLORS,
            category_orders={"Label": ["Good", "Bad"]},
            title=f"{algo} • {title_suffix}",
            width=800,
            height=800,
        )
        fig.update_layout(coloraxis_showscale=False, legend_title_text="Outcome")
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="black")))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Good/Bad Count")
        st.dataframe(make_individual_table(labels_bin, algo), use_container_width=True)

    # =============================== PORTFOLIO =================================
    else:  # vis_mode == "Portfolio"
        if data_src.startswith("Training"):
            y_best = load_from_cache("prelim_output.pkl").y_best    # (n_instances,) float
            df_viz = pd.DataFrame(
                {
                    "Z1": Z[:, 0],
                    "Z2": Z[:, 1],
                    "BestPerf": y_best,
                }
            )
            fig = px.scatter(
                df_viz,
                x="Z1",
                y="Z2",
                color="BestPerf",
                color_continuous_scale="Viridis",
                title="Portfolio • TRAINING (y_best)",
                width=800,
                height=800,
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.4, color="black")))
            st.plotly_chart(fig, use_container_width=True)

        else:  # Prediction (Selector)
            sel_variant = st.radio(
                "Selection variant",
                ["Primary (selection0)", "Fallback (selection1)"],
                horizontal=True,
            )
            selections = (
                output.selection0
                if sel_variant.startswith("Primary")
                else output.selection1
            )

            highlight = st.multiselect(
                "Highlight algorithms", algo_labels, default=algo_labels
            )
            color_map = {"Dimmed": "lightgrey", **algo_color_map}

            df_viz = pd.DataFrame(
                {
                    "Z1": Z[:, 0],
                    "Z2": Z[:, 1],
                    "Algo": [algo_labels[i] for i in selections],
                }
            )
            df_viz["Colour"] = [
                a if a in highlight else "Dimmed" for a in df_viz["Algo"]
            ]

            fig = px.scatter(
                df_viz,
                x="Z1",
                y="Z2",
                color="Colour",
                color_discrete_map=color_map,
                hover_data=["Algo"],
                title=f"Portfolio • PREDICTION – {sel_variant}",
                width=800,
                height=800,
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.4, color="black")))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Selection Counts (all algorithms)")
            st.dataframe(
                make_portfolio_table(
                    output.selection0, output.selection1, algo_labels
                ),
                use_container_width=True,
            )

    # --- summary ---------------------------------------------------------------
    st.subheader("📋 Summary of All Algorithm Performances")
    summary_df = output.pythia_summary.replace("", np.nan).infer_objects(copy=False)
    st.dataframe(summary_df, use_container_width=True)

    # --- download ZIP ----------------------------------------------------------
    st.subheader("📅 Download Cached PYTHIA Output")
    if cache_exists("pythia_output.pkl"):
        df_features = pd.DataFrame(output.w)
        df_probs = pd.DataFrame(output.pr0_hat, columns=algo_labels)
        instance_labels = getattr(
            load_from_cache("preprocessing_output.pkl"), "inst_labels", None
        )
        zip_bytes = create_stage_output_zip(
            x=df_features,
            y=df_probs,
            instance_labels=instance_labels,
            source_labels=None,
            metadata_description="PYTHIA stage cached output",
        )
        st.download_button(
            "⬇️ Download PYTHIA Output (ZIP)",
            zip_bytes,
            "pythia_output.zip",
            mime="application/zip",
        )
    else:
        st.warning("⚠️ No cached PYTHIA output found.")

    # --- cache management ------------------------------------------------------
    st.markdown("---")
    if st.button("❌ Delete PYTHIA Cache"):
        if delete_cache("pythia_output.pkl"):
            st.success("🗑️ PYTHIA cache deleted.")
        else:
            st.warning("⚠️ No PYTHIA cache file found to delete.")
