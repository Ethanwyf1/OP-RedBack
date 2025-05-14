import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import zipfile

from utils.cache_utils import load_from_cache, save_to_cache, delete_cache, cache_exists
from utils.download_utils import create_stage_output_zip
from utils.run_pilot import run_pilot


def show():
    st.header("📌 PILOT Stage – Dimensionality Reduction")

    if not cache_exists("sifted_output.pkl"):
        st.warning("⚠️ SIFTED output not found. Please run the SIFTED stage first.")
        return

    output = load_from_cache("sifted_output.pkl")

    # Attempt to load algorithm labels
    try:
        algo_labels = output.algo_labels
    except AttributeError:
        if cache_exists("preprocessing_output.pkl"):
            preprocessing_output = load_from_cache("preprocessing_output.pkl")
            try:
                algo_labels = preprocessing_output.algo_labels
            except AttributeError:
                algo_labels = [f"Algo {i}" for i in range(output.y_raw.shape[1])]
        else:
            algo_labels = [f"Algo {i}" for i in range(output.y_raw.shape[1])]

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

        pilot_output = run_pilot(
            x=output.x,
            y=output.y_raw[:, selected_index].reshape(-1, 1),
            feat_labels=output.feat_labels,
            analytic=analytic_mode,
            n_tries=n_tries
        )

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
        color_feat = st.selectbox("Color points by feature or performance:", ["None", selected_algo] + output.feat_labels)

        # === Plot ===
        st.subheader("📈 Instance Projection (Z Matrix)")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect("auto")

        if color_feat == selected_algo:
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

        ax.set_title("Instance Projection in 2D")
        ax.set_xlabel("Z1")
        ax.set_ylabel("Z2")
        st.pyplot(fig)

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
                        xticklabels=output.feat_labels, yticklabels=["Z1", "Z2"], ax=ax2)
            ax2.set_title("Projection Weight Heatmap")
            st.pyplot(fig2)

        # === Download Data ===
        st.subheader("📥 Download PILOT Output (Processed Data)")

        if cache_exists("pilot_output.pkl"):
            pilot_output = load_from_cache("pilot_output.pkl")

            # Projection Z matrix
            projection_df = pd.DataFrame(pilot_output.z, columns=["Z1", "Z2"])
            projection_csv = projection_df.to_csv(index_label="Instance")

            # A matrix (2 rows: Z1 and Z2, columns: features)
            weights_df = pd.DataFrame(
                pilot_output.a,
                columns=output.feat_labels,
                index=["Z1", "Z2"]
            )
            weights_csv = weights_df.to_csv(index=True)

            # README
            readme_text = (
                "This ZIP archive contains the output from the PILOT stage:\n\n"
                "- Z_matrix.csv: The 2D projection of instances.\n"
                "- Feature_Weights_A.csv: Projection matrix A (rows = Z1, Z2).\n"
                "- README.txt: This file.\n"
            )

            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("Z_matrix.csv", projection_csv)
                zip_file.writestr("Feature_Weights_A.csv", weights_csv)
                zip_file.writestr("README.txt", readme_text)

            zip_buffer.seek(0)

            # Streamlit download button
            st.download_button(
                label="⬇️ Download PILOT Output (ZIP)",
                data=zip_buffer,
                file_name="pilot_output.zip",
                mime="application/zip"
            )
        else:
            st.warning("⚠️ No PILOT cache found. Please run the stage first.")

        # === Cache Management ===
        st.subheader("🗑️ Delete PILOT Cache")

        if st.button("❌ Delete PILOT Output Cache"):
            success = delete_cache("pilot_output.pkl")
            if success:
                st.success("✅ PILOT cache deleted successfully.")
                st.session_state.pop("pilot_output", None)
                st.session_state["ran_pilot"] = False
            else:
                st.warning("⚠️ No PILOT cache found to delete.")

