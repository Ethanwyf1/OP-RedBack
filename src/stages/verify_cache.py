import streamlit as st

from utils.cache_utils import cache_exists, load_from_cache


def show():
    st.title("🔎 Verify Preprocessing Cache")

    if cache_exists("preprocessing_output.pkl"):
        st.success("✅ Cache file exists.")

        output = load_from_cache("preprocessing_output.pkl")

        st.markdown("### 📊 Output Summary")
        st.write(f"Number of instances: {output.x.shape[0]}")
        st.write(f"Number of selected features: {len(output.feat_labels)}")
        st.write(f"Number of selected algorithms: {len(output.algo_labels)}")

        st.markdown("### 🧬 Feature Labels")
        st.write(output.feat_labels[:10])  # show first 10 features

        st.markdown("### ⚙️ Algorithm Labels")
        st.write(output.algo_labels[:10])  # show first 10 algorithms

        st.markdown("### 📁 Instance Labels Preview")
        st.dataframe(output.inst_labels.head())

    else:
        st.error("❌ preprocessing_output.pkl not found. Please run Preprocessing first.")
