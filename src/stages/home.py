import streamlit as st
import pandas as pd
import os

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

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        with open("cache/uploaded_metadata.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_file_path"] = "cache/uploaded_metadata.csv"

        df = pd.read_csv("cache/uploaded_metadata.csv")
        st.success("✅ File uploaded and cached!")
        st.dataframe(df.head())

    elif os.path.exists("cache/uploaded_metadata.csv"):
        st.info("📌 Using previously uploaded metadata file.")
        df = pd.read_csv("cache/uploaded_metadata.csv")
        st.session_state["uploaded_file_path"] = "cache/uploaded_metadata.csv"
        st.dataframe(df.head())
    else:
        st.warning("⚠️ Please upload your metadata file to continue.")

    if os.path.exists("cache/uploaded_metadata.csv"):
        if st.button("🗑️ Delete Uploaded Metadata"):
            os.remove("cache/uploaded_metadata.csv")
            st.session_state.pop("uploaded_file", None)
            st.warning("Uploaded metadata has been deleted.")
