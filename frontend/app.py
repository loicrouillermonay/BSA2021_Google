import streamlit as st
import joblib
import os
import pandas as pd


def main():
    """Lingorank frontend"""

    st.title("Lingorank UI")
    st.subheader("BSA2021 - Team Google")

    # Creating sidebar with selection box
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown("Some information here")
        st.subheader("This is a subheader")

    # Predication page
    if selection == "Prediction":
        st.info("Prediction with Team Google's ML Models")
        # Creating a text box for user input
        sentence = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # do some stuff here
            st.success("Text Difficulty categorized as: B2")


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
