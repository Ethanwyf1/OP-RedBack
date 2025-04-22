import streamlit as st
from utils.cache_utils import save_to_cache, load_from_cache, delete_cache, cache_exists

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
        prelim_output = load_from_cache("prelim_output.pkl")
        st.success("✅ Prelimed data loaded successfully!")

        