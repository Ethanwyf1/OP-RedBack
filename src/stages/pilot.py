import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

from utils.cache_utils import load_from_cache, save_to_cache, delete_cache, cache_exists
from instancespace.stages.pilot import PilotStage, PilotInput
from instancespace.data.options import PilotOptions

def show():
    st.header("📌 PILOT Stage – Dimensionality Reduction")

    if not cache_exists("preprocessing_output.pkl"):
        st.warning("⚠️ Preprocessing output not found. Please run the Preprocessing stage first.")
        return

    output = load_from_cache("preprocessing_output.pkl")
    algo_labels = output.algo_labels
    inst_labels = getattr(output, "inst_labels", [f"Inst {i}" for i in range(len(output.x))])

    st.markdown("This stage projects high-dimensional instances into a 2D space for visualization and later analysis.")

    # === Projection Settings ===
    st.subheader("⚙️ Projection Settings")
    selected_algo = st.selectbox("Choose algorithm performance to project by:", algo_labels)
    selected_index = algo_labels.index(selected_algo)

    col1, col2 = st.columns(2)
    with col1:
        analytic_mode = st.checkbox("Use Analytic Mode (PCA-like)", value=True)
    with col2:
        n_tries = st.number_input("BFGS Trials (Only used if not Analytic)", value=10, min_value=1)

    if st.button("🚀 Run PILOT"):
        st.info("Running PILOT...")

        pilot_input = PilotInput(
            x=output.x,
            y=output.y_raw[:, selected_index].reshape(-1, 1),
            feat_labels=output.feat_labels,
            pilot_options=PilotOptions(
                x0=None,
                alpha=None,
                analytic=analytic_mode,
                n_tries=n_tries
            )
        )

        pilot_output = PilotStage._run(pilot_input)
        save_to_cache(pilot_output, "pilot_output.pkl")
        st.session_state["pilot_output"] = pilot_output
        st.session_state["ran_pilot"] = True
        st.toast("PILOT completed successfully!", icon="✅")

    if st.session_state.get("ran_pilot", False) or cache_exists("pilot_output.pkl"):
        pilot_output = st.session_state.get("pilot_output") or load_from_cache("pilot_output.pkl")

        Z = pilot_output.z
        A = pilot_output.a
        summary_df = pilot_output.pilot_summary
        error = pilot_output.error
        r2 = pilot_output.r2

        # === Coloring Options ===
        st.subheader("🎨 Coloring Options")
        color_feat = st.selectbox("Color points by feature or performance:", ["None", "Z₁", "Z₂", selected_algo] + output.feat_labels)

        # === Plot ===
        st.subheader("📈 Instance Projection (Z Matrix)")
        show_hull = st.checkbox("Show Convex Hull Boundary", value=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect("auto")

        if color_feat == "Z₁":
            color_vals = Z[:, 0]
            label = "Z₁"
        elif color_feat == "Z₂":
            color_vals = Z[:, 1]
            label = "Z₂"
        elif color_feat == selected_algo:
            color_vals = output.y_raw[:, selected_index]
            label = f"{selected_algo} Performance"
        elif color_feat in output.feat_labels:
            color_vals = output.x[:, output.feat_labels.index(color_feat)]
            label = color_feat
        else:
            color_vals = None
            label = ""

        if color_vals is not None:
            scatter = ax.scatter(Z[:, 0], Z[:, 1], c=color_vals, cmap="viridis", alpha=0.3, s=10, edgecolors="none")
            plt.colorbar(scatter, label=label)
        else:
            ax.scatter(Z[:, 0], Z[:, 1], color="royalblue", alpha=0.3, s=10, edgecolors="none")

        if show_hull:
            hull = ConvexHull(Z)
            for simplex in hull.simplices:
                ax.plot(Z[simplex, 0], Z[simplex, 1], "r--", linewidth=1)

        ax.set_title("Instance Projection in 2D")
        ax.set_xlabel("Z₁")
        ax.set_ylabel("Z₂")
        ax.set_aspect("auto")
        st.pyplot(fig)

        # === Export Z ===
        st.download_button(
            "⬇️ Download Z Matrix (Projection Results)",
            data=pd.DataFrame(Z, columns=["Z1", "Z2"]).to_csv(index=False),
            file_name="pilot_projection.csv",
            mime="text/csv"
        )

        # === Projection Metrics ===
        st.metric("Projection Error", round(error, 4))
        st.metric("Mean R²", round(np.mean(r2), 4))

        # === Matrix A ===
        st.subheader("📊 Projection Matrix A (Feature Weights)")
        with st.expander("🔬 View A Matrix as Table"):
            st.dataframe(summary_df.set_index(0), use_container_width=True)

        with st.expander("🔥 Heatmap of A Matrix"):
            fig2, ax2 = plt.subplots(figsize=(10, 2))
            sns.heatmap(A, annot=True, cmap="coolwarm", cbar=True,
                        xticklabels=output.feat_labels, yticklabels=["Z₁", "Z₂"], ax=ax2)
            ax2.set_title("Projection Weight Heatmap")
            st.pyplot(fig2)

        # === Cache Controls ===
        st.subheader("🗑️ Cache Management")
        if st.button("❌ Delete PILOT Cache"):
            success = delete_cache("pilot_output.pkl")
            if success:
                st.success("✅ PILOT cache deleted.")
            else:
                st.warning("⚠️ No cache file found.")
