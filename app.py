import streamlit as st

st.set_page_config(page_title="Instance Space Visualizer", layout="wide")
st.title("🧠 Instance Space Visualizer")

st.sidebar.header("Select Stage")
stage = st.sidebar.radio("Stage", [
    "Preprocessing", "PRELIM", "SIFTED", "PILOT", "CLOISTER", "PYTHIA", "TRACE"
])

# Import the display module for each stage
if stage == "Preprocessing":
    from stages import preprocessing
    preprocessing.show()

elif stage == "PRELIM":
    from stages import prelim
    prelim.show()

elif stage == "SIFTED":
    from stages import sifted
    sifted.show()

elif stage == "PILOT":
    from stages import pilot
    pilot.show()

elif stage == "CLOISTER":
    from stages import cloister
    cloister.show()

elif stage == "PYTHIA":
    from stages import pythia
    pythia.show()

elif stage == "TRACE":
    from stages import trace
    trace.show()
