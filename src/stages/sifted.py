import io
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from instancespace.data.options import (ParallelOptions, SelvarsOptions,
                                        SiftedOptions)
from instancespace.stages.sifted import SiftedInput, SiftedStage
from sklearn.decomposition import PCA

from utils.cache_utils import (cache_exists, delete_cache, load_from_cache,
                               save_to_cache)


def display_sifted_output(sifted_out, preprocessing_output):
    if sifted_out.rho is None:
        st.warning(
            "⚠️ Not enough features selected. The SIFTED stage skipped correlation analysis."
        )
        return

    """
    Visualization：
      1) Feature Importance
      2) Correlation Heatmap（feature vs algorithm)
      3) Silhouette curve
      4) Clust tale
    """

    rho_full = np.abs(sifted_out.rho)  # shape = (all_feats, n_algos)
    sel = sifted_out.selvars
    rho_sel = rho_full[sel, :]  # shape = (n_sel_feats, n_algos)
    feat_labels = sifted_out.feat_labels
    algo_names = preprocessing_output.algo_labels

    # —— 1. Feature Importance ——
    st.subheader("📊 Selected Feature Importance")
    with st.expander("❓ What is this chart?"):
        st.write(
            "Each bar shows the maximum absolute Pearson correlation of that feature "
            "with any algorithm. It highlights which features have the strongest predictive signal."
        )

    if rho_sel.size:
        feat_imp = rho_sel.max(axis=1)  # choose max |ρ| in each selected feature
        df_imp = pd.DataFrame({"importance": feat_imp}, index=feat_labels)
        st.bar_chart(df_imp)
        # downlod the image
        fig, ax = plt.subplots()
        df_imp.plot.bar(ax=ax)
        ax.set_ylabel("Max |ρ|")
        ax.set_title("Selected Feature Importance")
        plt.tight_layout()

        # 1) Turn it into a PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        # 2) Offer that buffer as a download
        st.download_button(
            label="📥 Download Plot as PNG",
            data=buf,
            file_name="feature_importance.png",
            mime="image/png",
        )
    else:
        st.info("No selected features to plot importance.")

    # —— 2. Correlation Heatmap ——
    st.subheader("🗺️ Weights for features")
    with st.expander("❓ What is this chart?"):
        st.write(
            "p is the array of p-values coming out of your feature-vs-performance correlation tests. "
        )
    if rho_sel.size:
        fig, ax = plt.subplots(figsize=(6, max(3, len(feat_labels) * 0.3)))
        im = ax.imshow(rho_sel, aspect="auto", cmap="viridis")

        ax.set_yticks(np.arange(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=8)

        ax.set_xticks(np.arange(len(algo_names)))
        ax.set_xticklabels(algo_names, rotation=90, fontsize=8)
        fig.colorbar(im, ax=ax, label="|ρ|")
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        # Download button
        st.download_button(
            label="📥 Download Weight Heatmap",
            data=buf,
            file_name="Weights_for_features.png",
            mime="image/png",
        )
    else:
        st.info("No correlation matrix to display.")

    # —— 3. Silhouette score ——
    st.subheader("📈 Silhouette Scores vs. #Clusters")
    if sifted_out.silhouette_scores:
        df_sil = pd.DataFrame(
            {"silhouette_score": sifted_out.silhouette_scores},
            index=range(2, 2 + len(sifted_out.silhouette_scores)),
        )
        st.line_chart(df_sil)
    else:
        st.info("No silhouette scores provided.")


    if sifted_out.clust is not None:

        clust = sifted_out.clust
        n_clusters = clust.shape[1]
        col_names = [f"cluster{i+1}" for i in range(n_clusters)]

        df = pd.DataFrame(clust, columns=col_names)
        
        st.subheader("Cluster Boolean Table")
        st.dataframe(df)

        selected_features = sifted_out.feat_labels
        sel_df = pd.DataFrame(selected_features, columns=["Selected Features"])
        st.subheader("Selected Features")
        st.dataframe(sel_df)

    else:
        st.info("No Cluster Info Available.")

    # # —— 4. PCA 2D projection —— 
    # st.subheader("🔬 Instances in 2D (PCA Projection)")
    # if sifted_out.x.shape[1] >= 2:
    #     pca = PCA(n_components=2, random_state=0)
    #     proj = pca.fit_transform(sifted_out.x)
    #     fig2, ax2 = plt.subplots(figsize=(5,5))
    #     ax2.scatter(proj[:,0], proj[:,1], s=15, alpha=0.6)
    #     ax2.set_xlabel("PC 1")
    #     ax2.set_ylabel("PC 2")
    #     ax2.set_title("PCA of Sifted Features")
    #     st.pyplot(fig2)
    # else:
    #     st.info("Not enough features for 2D PCA.")

def show():
    st.header("🧹 SIFTED Stage")
    st.write("This is a placeholder for the SIFTED stage visualization.")

    if not cache_exists("prelim_output.pkl"):
        st.error("🚫 Prelim output not found. Please run the Prelim stage first.")
        if st.button("Go to Prelim Stage"):
            st.session_state.current_tab = "prelim"
            st.experimental_rerun()
        return
    else:
        preprocessing_output = load_from_cache("preprocessing_output.pkl")
        prelim_output = load_from_cache("prelim_output.pkl")
        st.success("✅ Prelimed data loaded successfully!")

    with st.form("sifted_config_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("General & Clustering Settings")

            flag = st.checkbox(
                "Enable density‑based filtering",
                value=True,
                help="When True, instances will be filtered based on density/minimum distance criteria.",
            )

            rho = st.number_input(
                "Correlation threshold (rho)",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.01,
                help="Minimum |Pearson correlation| between a feature and any algorithm to keep that feature.",
            )

            k = st.number_input(
                "Number of clusters (k)",
                min_value=1,
                value=5,
                step=1,
                help="Number of clusters for KMeans-based feature grouping. If #features ≤ k, clustering is skipped.",
            )
            n_trees = st.number_input(
                "Number of trees (n_trees)",
                min_value=1,
                value=100,
                step=1,
                help="Number of trees used in any random‑forest step (if applicable).",
            )
            max_iter = st.number_input(
                "KMeans max iterations",
                min_value=1,
                value=300,
                step=1,
                help="Maximum iterations allowed for the KMeans clustering algorithm.",
            )
            replicates = st.number_input(
                "KMeans replicates (n_init)",
                min_value=1,
                value=10,
                step=1,
                help="Number of times to run KMeans with different centroid seeds to choose the best.",
            )

        with col2:
            st.subheader("Genetic Algorithm Settings")
            num_generations = st.number_input(
                "GA generations",
                min_value=1,
                value=50,
                step=1,
                help="Total number of generations to evolve in the genetic algorithm.",
            )
            sol_per_pop = st.number_input(
                "Population size",
                min_value=1,
                value=50,
                step=1,
                help="Number of candidate solutions (individuals) in each GA generation.",
            )
            num_parents_mating = st.number_input(
                "Parents per mating",
                min_value=1,
                value=20,
                step=1,
                help="How many parents are selected to produce offspring each generation.",
            )
            parent_selection_type = st.selectbox(
                "Parent selection type",
                options=["tournament", "roulette"],
                help="Strategy to choose parents for crossover: 'tournament' or 'roulette wheel'.",
            )
            k_tournament = st.number_input(
                "Tournament size",
                min_value=1,
                value=3,
                step=1,
                help="Number of individuals competing in each tournament if using tournament selection.",
            )
            keep_elitism = st.number_input(
                "Elitism count",
                min_value=0,
                value=2,
                step=1,
                help="Number of best solutions automatically carried over to the next generation.",
            )
            crossover_type = st.selectbox(
                "Crossover operator",
                options=["single_point", "two_point", "uniform"],
                help="Type of crossover to use when combining two parent solutions.",
            )
            cross_over_probability = st.number_input(
                "Crossover probability",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.01,
                help="Probability that crossover will occur between two parents.",
            )
            mutation_type = st.selectbox(
                "Mutation operator",
                options=["random", "swap"],
                help="Type of mutation applied to offspring: random gene changes or swaps.",
            )
            mutation_probability = st.number_input(
                "Mutation probability",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                help="Chance that each gene in an individual will be mutated.",
            )
            stop_criteria = st.selectbox(
                "GA stop criteria",
                options=["saturate_5", "reach_0"],
                help="When to stop the GA: 'saturate_5' (no improvement for 5 generations), 'reach_0' (fitness reaches zero).",
            )

        submitted = st.form_submit_button("Run Sifted Stage")

    if submitted:

        try:
            sifted_opts = SiftedOptions(
                flag=flag,
                rho=rho,
                k=k,
                n_trees=n_trees,
                max_iter=max_iter,
                replicates=replicates,
                num_generations=num_generations,
                num_parents_mating=num_generations,
                sol_per_pop=sol_per_pop,
                parent_selection_type=parent_selection_type,
                k_tournament=k_tournament,
                keep_elitism=keep_elitism,
                crossover_type=crossover_type,
                cross_over_probability=cross_over_probability,
                mutation_type=mutation_type,
                mutation_probability=mutation_probability,
                stop_criteria=stop_criteria,
            )

            selvars_opts = SelvarsOptions(
                feats=None,
                algos=None,
                small_scale_flag=False,
                small_scale=0.1,
                file_idx_flag=False,
                file_idx="",
                selvars_type="auto",
                min_distance=0.0,
                density_flag=False,
            )

            parallel_opts = ParallelOptions(flag=False, n_cores=20)

            sifted_in = SiftedInput(
                x=prelim_output.x,
                y=prelim_output.y,
                y_bin=prelim_output.y_bin,
                x_raw=prelim_output.x_raw,
                y_raw=prelim_output.y_raw,
                beta=prelim_output.beta,
                num_good_algos=prelim_output.num_good_algos,
                y_best=prelim_output.y_best,
                p=prelim_output.p,
                inst_labels=prelim_output.instlabels,
                feat_labels=preprocessing_output.feat_labels,
                s=prelim_output.s,
                sifted_options=sifted_opts,
                selvars_options=selvars_opts,
                data_dense=prelim_output.data_dense,
                parallel_options=parallel_opts,
            )

            st.info(
                f"📐 Prelim feature count: {prelim_output.x.shape[1]} features available before SIFTED."
            )

            sifted_output = SiftedStage._run(sifted_in)

            if sifted_output is not None:

                # Save to session state and cache
                st.session_state["sifted_output"] = sifted_output
                st.session_state["ran_sifted"] = True
                save_to_cache(sifted_output, "sifted_output.pkl")

                # Show success message
                st.toast("✅ Sifted stage run successfully!", icon="🚀")

        except Exception as e:
            st.error(f"Error running Sifted stage: {str(e)}")
            st.code(traceback.format_exc())

    if st.session_state.get("ran_sifted", False):
        try:
            
            if "sifted_output" not in st.session_state and cache_exists("sifted_output.pkl"):
                st.session_state["sifted_output"] = load_from_cache("sifted_output.pkl")
                st.session_state["ran_sifted"] = True

            sifted_output = st.session_state["sifted_output"]
            display_sifted_output(sifted_output, preprocessing_output)

            st.subheader("🗑️ Cache Management")
            if st.button("❌ Delete Sifted Cache"):
                success = delete_cache("sifted_output.pkl")
                if success:
                    st.success("🗑️ Sifted cache deleted.")
                    if "sifted_output" in st.session_state:
                        del st.session_state["sifted_output"]
                    if "ran_sifted" in st.session_state:
                        del st.session_state["ran_sifted"]
                else:
                    st.warning("⚠️ No cache file found to delete.")

        except Exception as e:
            st.error(f"Error displaying Sifted results: {str(e)}")
            st.code(traceback.format_exc())
