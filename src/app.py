import streamlit as st

st.set_page_config(page_title="Instance Space Visualizer", layout="wide")
st.title("🧠 Instance Space Visualizer")

# 1. Define available stages
stage_names = [
    "Home",
    "Preprocessing",
    "PRELIM",
    "SIFTED",
    "PILOT",
    "CLOISTER",
    "PYTHIA",
    "TRACE",
    "Cache",
]

# 2. Read stage from query param or default to Home
params = st.query_params
default_stage = params.get("stage", ["Home"])[0]
if default_stage not in stage_names:
    default_stage = "Home"

# 3. Sidebar selection
selected_stage = st.sidebar.radio(
    "Select Stage", stage_names, index=stage_names.index(default_stage)
)

# 4. Update query param if changed
if selected_stage != default_stage:
    st.query_params.update(stage=selected_stage)

# 5. Load the corresponding stage module
if selected_stage == "Home":
    from stages import home

    home.show()

elif selected_stage == "Preprocessing":
    from stages import preprocessing

    preprocessing.show()

elif selected_stage == "PRELIM":
    from stages import prelim

    prelim.show()

elif selected_stage == "SIFTED":
    from stages import sifted

    sifted.show()

elif selected_stage == "PILOT":
    from stages import pilot

    pilot.show()

elif selected_stage == "CLOISTER":
    from stages import cloister

    cloister.show()

elif selected_stage == "PYTHIA":
    from stages import pythia

    pythia.show()

elif selected_stage == "TRACE":
    from stages import trace

    trace.show()

elif selected_stage == "Cache":
    from stages import verify_cache

    verify_cache.show()
