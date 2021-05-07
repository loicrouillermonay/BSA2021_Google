import streamlit as st
import requests


def streamlit_config():
    st.set_page_config(
        page_title='Lingorank')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            span {border-radius: 8px}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def main():
    """BSA 2021: Team Google - Lingorank frontend"""
    streamlit_config()
    st.markdown("# Lingorank UI v2.0")
    st.markdown("### Lingorank predicts the difficulty of a French written sentence.")
    st.markdown(" Enter your French sentence below")

    # Creating a text box for user input
    sentence = st.text_area(
        "", "Remplacez cette phrase par celle que vous voulez faire analyser.")

    # Create the button
    if st.button("Classify"):
        # On click, display waiting message while querying
        with st.spinner(text='Processing overall sentence difficulty...'):
            # Query request and response with overall difficulty
            query = {'text': sentence}
            response = requests.get(
                'http://51.103.167.205/api/predict', params=query)
                # Display the result of the query
            st.success(
                f"Succesfully categorized overall difficulty as: **{response.json()['difficulty']}**")
        # After the success of the previous operation, perform a second query with individual words difficulty
        with st.spinner(text='Processing individual words difficulty'):
            response_diff = requests.get(
                'http://51.103.167.205/api/predict/words', params=query)
            # Assign list of individual difficulty to a variable
            difficulties = response_diff.json()['difficulty']

            # Some layout
            caption = '''<hr><h4>Find here the individual difficulty of words. Background color varies depending on the level.</h4> 
            <i>NOTA BENE :</i> Level is usually low because words are analyzed without their context 
            <br> <br>
            <div style="text-align:center"> Caption : 
            <span style = 'background-color:#2E541A'>[ A1 ]</span> - 
            <span style = 'background-color:#6C8A1A'>[ A2 ]</span> - 
            <span style = 'background-color:#F7C32D'>[ B1 ]</span> - 
            <span style = 'background-color:#DE700F'>[ B2 ]</span> - 
            <span style = 'background-color:#B21F01'>[ C1 ]</span> - 
            <span style = 'background-color:#7B1200'>[ C2 ]</span>
            </div> <br>
            '''
            
            # Prepare variables to display the sentence with individual difficulties
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
            # Display caption and sentence with indificual difficulties
            st.markdown(caption, unsafe_allow_html=True)
            output = "<i>" + output + "</i>"
            st.markdown(output, unsafe_allow_html=True)
    

    


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
