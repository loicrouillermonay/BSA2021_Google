import streamlit as st
import requests


def streamlit_config():
    st.set_page_config(
        page_title='Lingorank')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def main():
    """BSA 2021: Team Google - Lingorank frontend"""
    streamlit_config()
    st.title("Lingorank UI v2.0")
    st.text(" Lingorank predicts the difficulty of a French written sentence.")

    # Creating a text box for user input
    sentence = st.text_area(
        "Enter a French written sentence.", "Entrez votre phrase ici.")

    if st.button("Classify"):
        with st.spinner(text='In progress...'):
            query = {'text': sentence}
            response = requests.get(
                'http://51.103.169.80/api/predict', params=query)
            st.success(
                f"Difficulty categorized as: {response.json()['difficulty']}")


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
