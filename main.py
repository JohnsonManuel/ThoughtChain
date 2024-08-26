import streamlit as st
import youtube_helper as yth
import textwrap

if st.secrets["HUGGINGFACEHUB_API_TOKEN"] != "" :
    st.session_state["api_key"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None

# Sidebar content
with st.sidebar:
    if st.session_state['api_key'] is None:
        # First form to input API key
        form1 = st.form(key="api_key_form")
        with form1:
            st.title("Enter API Key")
            api_key_input = st.text_input("API Key", type="password")
            submit_button = st.form_submit_button("Submit")

            # Save the API key if the form is submitted
            if submit_button and api_key_input != "":
                st.session_state['api_key'] = api_key_input
                st.rerun()

    else:
        # Display new form once API key is entered
        with st.form(key='secondary_form'):
            st.title("Ask questions on any youtube video!")
            
            youtube_url = st.text_area(
                label="What is the YouTube video URL?",
            )
            query = st.text_area(
                label="Ask me about the video?",
                key="query"
            )
            submit_button = st.form_submit_button("Submit")

            if submit_button and youtube_url and query:
                db = yth.create_db_from_youtube_video_url(youtube_url)
                response, docs = yth.get_response_from_query(db, query, st.session_state['api_key'])
                st.session_state['response'] = response

# Main page content
st.title("Youtube Assistant")

if st.session_state['api_key']:
    st.sidebar.success("Your API key will be used for the App.")
else:
    st.sidebar.warning("Please enter your Hugging Face API key to continue.")

# Display the answer and other details in the main body
if 'response' in st.session_state:
    st.subheader("Answer:")
    st.text(textwrap.fill(st.session_state['response'], width=85))
