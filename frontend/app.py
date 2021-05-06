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
    st.markdown("# Lingorank UI v2.0")
    st.markdown("### Lingorank predicts the difficulty of a French written sentence.")
    st.markdown(" Enter your French sentence below")

    difficulties = [0,0,0,0,0,0,1,1,1,0,2,3,4,5,0,1,2,3,4,5]

    # Creating a text box for user input
    sentence = st.text_area(
        "", "Remplacez cette phrase par celle que vous voulez faire analyser. Remplacez cette phrase par celle que vous voulez faire.")

    if st.button("Classify"):
        with st.spinner(text='In progress...'):
            query = {'text': sentence}
            response = requests.get(
                'http://51.103.141.180/api/predict', params=query)
            #st.success(
            #    f"Difficulty categorized as: {response.json()['difficulty']}")

    st.markdown("---")

    index = 0
    bloc = sentence.split(' ')
    output = ""
    for i in bloc:
        if(difficulties[index] == 0):
            output = output + "<span style = 'background-color:#2E541A'>" + bloc[index] + "</span>" + " "
        if(difficulties[index] == 1):
            output = output + "<span style = 'background-color:#6C8A1A'>" + bloc[index] + "</span>" + " "
        if(difficulties[index] == 2):
            output = output + "<span style = 'background-color:#F7C32D'>" + bloc[index] + "</span>" + " "
        if(difficulties[index] == 3):
            output = output + "<span style = 'background-color:#DE700F'>" + bloc[index] + "</span>" + " "
        if(difficulties[index] == 4):
            output = output + "<span style = 'background-color:#B21F01'>" + bloc[index] + "</span>" + " "
        if(difficulties[index] == 5):
            output = output + "<span style = 'background-color:#7B1200'>" + bloc[index] + "</span>" + " "

        index = index + 1
    st.markdown(output, unsafe_allow_html=True)
    

    


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
