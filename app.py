import streamlit as st
import Home
import About

# Dictionary to map page names to page modules
pages = {
    "Demo": Home,
    "Details": About
}

# Streamlit sidebar for navigation
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Load the selected page
page = pages[selection]
page.app()
