import streamlit as st
import pandas as pd

def show():
    st.header("🏠 Welcome to the Instance Space Visualizer")

    st.markdown("""
    This platform provides an interactive environment for conducting **Instance Space Analysis (ISA)** —  
    a data-driven method to evaluate algorithm performance across diverse instances.

    - 📦 Upload your metadata and configuration files  
    - 🔍 Explore feature and algorithm relationships  
    - 🧠 Visualize performance footprints   
    """)

    st.markdown("### 🔁 ISA Pipeline Overview")
    st.markdown("""
    1. **Preprocessing** – Clean and normalize your data  
    2. **PRELIM** – Label good/bad performance using thresholds  
    3. **SIFTED** – Select relevant features  
    4. **PILOT** – Reduce dimensionality  
    5. **CLOISTER** – Define instance space boundaries  
    6. **PYTHIA** – Predict the best algorithm for new instances  
    7. **TRACE** – Visualize algorithm footprints and evaluate purity  
    """)

    st.info("📌 Select a stage from the sidebar to begin.")

    ##Upload file
    st.markdown("### 📂 Upload Your Metadata File (.csv)")

    uploaded_file = st.file_uploader("Upload a CSV file containing instance features and performance data", type=["csv"])

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file 
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head(), use_container_width=True)
            st.caption("📌 This is a preview of the uploaded dataset. You can proceed to the Preprocessing stage.")
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
