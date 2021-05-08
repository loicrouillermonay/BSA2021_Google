import streamlit as st
import requests


def streamlit_config():
    st.set_page_config(
        page_title='Lingorank')
    # In-page style
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            span {border-radius: 8px;}
            .dark-green {background-color: green;}
            .light-green {background-color: #6C8A1A;}
            .yellow {background-color: #F7C32D;}
            .orange {background-color: #DE700F;}
            .red {background-color: #B21F01;}
            .dark-red {background-color: red;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def main():
    """BSA 2021: Team Google - Lingorank frontend"""
    streamlit_config()
    st.markdown("# Lingorank Web App")
    st.markdown(
        "### Lingorank predicts the difficulty of a French written sentence.")
    st.markdown(" Enter your French sentence below")

    # Creating a text box for user input
    sentence = st.text_area(
        "", "Remplacez cette phrase par celle que vous voulez faire analyser.")

    api = "http://51.103.167.205/api/"

    # Create the button
    if st.button("Classify"):
        # On click, display waiting message while querying
        with st.spinner(text='Processing overall sentence difficulty...'):
            # Query request and response with overall difficulty
            query = {'text': sentence}
            response = requests.get(
                api+'predict', params=query)
            # Display the result of the query
            st.success(
                f"Succesfully categorized overall difficulty as: **{response.json()['difficulty']}**")
        # After the success of the previous operation, perform a second query with individual words difficulty
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.spinner(text='Processing individual words difficulty'):
            response_diff = requests.get(
                api+'predict/words', params=query)
            # Assign list of individual difficulty to a variable
            difficulties = response_diff.json()['difficulty']

            # Some layout
            caption = '''<h4>Find here the individual difficulty of words. Background color varies depending on the level.</h4> 
            <i>NOTA BENE :</i> Difficulty levels of words are usually low and far from the predicted difficulty of the sentence because this process does not take into account the relationship between the words.
            <br> <br>
            <div style="text-align:center"> Label : 
            <span class = 'dark-green'>[ A1 ]</span> - 
            <span class = 'light-green'>[ A2 ]</span> - 
            <span class = 'yellow'>[ B1 ]</span> - 
            <span class = 'orange'>[ B2 ]</span> - 
            <span class = 'red'>[ C1 ]</span> - 
            <span class = 'dark-red'>[ C2 ]</span>
            </div> <br>
            '''

            # Prepare variables to display the sentence with individual difficulties
            index = 0
            bloc = sentence.split(' ')
            output = ""
            for i in bloc:
                if(difficulties[index] == 0):
                    output = output + "<span class = 'dark-green'>" + \
                        bloc[index] + " </span>"
                if(difficulties[index] == 1):
                    output = output + "<span class = 'light-green'>" + \
                        bloc[index] + " </span>"
                if(difficulties[index] == 2):
                    output = output + "<span class = 'yellow'>" + \
                        bloc[index] + " </span>"
                if(difficulties[index] == 3):
                    output = output + "<span class = 'orange'>" + \
                        bloc[index] + " </span>"
                if(difficulties[index] == 4):
                    output = output + "<span class = 'red'>" + \
                        bloc[index] + " </span>"
                if(difficulties[index] == 5):
                    output = output + "<span class = 'dark-red'>" + \
                        bloc[index] + " </span>"

                index = index + 1
            # Display caption and sentence with indificual difficulties
            st.markdown(caption, unsafe_allow_html=True)
            output = "<i>" + output + "</i>"
            st.markdown(output, unsafe_allow_html=True)


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
