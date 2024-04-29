import streamlit as st
import requests

# Define API endpoints
API_BASE_URL = "http://127.0.0.1:8000"  # Change this to your actual API server URL

# Function to call API for updating settings and loading/building index
def update_settings_and_load_or_build_index(api_key, model, file):
    # Prepare the file in the correct format for the multipart/form-data POST request
    files = {
        'file': (file.name, file, 'multipart/form-data')
    }
    
    # Prepare the rest of the form data
    data = {
        'cohere_api_key': api_key,
        'cohere_model': model
    }

    # Post the data to the FastAPI endpoint
    response = requests.post(
        f"{API_BASE_URL}/update_settings_and_load_or_build_index",
        files=files,
        data=data
    )

    # Check the response
    if response.status_code == 200:
        # If the response is OK, return True and any message from the API
        return True, response.json()
    else:
        # If there was an error, return False and the error message
        return False, response.text

# Functions to interact with your API
def get_response(query):
    response = requests.post(f"{API_BASE_URL}/response", json={"query": query})
    if response.ok:
        return response.text
    else:
        return "Error: " + response.text

def stream_response(query):
    response = requests.post(f"{API_BASE_URL}/stream", json={"query": query}, stream=True)
    if response.ok:
        # Stream response chunk by chunk
        for chunk in response.iter_content(chunk_size=10):  # Adjust chunk_size as needed
            if chunk:  # Filter out keep-alive new chunks
                yield chunk.decode('utf-8')
    else:
        return None


def query_index(query):
    response = requests.post(f"{API_BASE_URL}/query", json={"query": query})
    if response.ok:
        return response.text
    else:
        return "Error: " + response.text

# Initialize session state variable for settings_updated
if 'settings_updated' not in st.session_state:
    st.session_state['settings_updated'] = False

# Sidebar for settings and file upload
with st.sidebar:
    st.title("Settings and Index Update")
    api_key = st.text_input("Enter Cohere API Key:", type="password")
    model = "command-r-plus"
    file = st.file_uploader("Upload file to build index from:", type=['pdf'])

    if st.button("Update Settings and Build Index"):
        if api_key and model and file is not None:
            success, message = update_settings_and_load_or_build_index(api_key, model, file)
            if success:
                st.success("Settings updated and index built successfully.")
                st.session_state['settings_updated'] = True
            else:
                st.error(f"Failed to update settings and build index: {message}")
        else:
            st.error("Please provide API key, model, and file.")

# Main application
if st.session_state['settings_updated']:
    st.title("Query Interface")

    # Radio button selection for operation
    operation = st.radio("Select an Operation:", ["Response", "Streaming Response"])

    # Response Operation
    if operation == "Response":
        with st.form(key='ResponseForm'):
            response_query = st.text_input("Enter your query for a response:", key="response_query")
            submit_response_query = st.form_submit_button("Get Response")
        if submit_response_query and response_query:
            response_result = get_response(response_query)
            st.text_area("Response Result", value=response_result, height=300)

    elif operation == "Streaming Response":
        with st.form(key='StreamingForm'):
            stream_query = st.text_input("Enter your query for streaming:", key="stream_query")
            submit_stream_query = st.form_submit_button("Start Streaming")
        if submit_stream_query and stream_query:
            st.write("Streaming responses:")
            chunks = stream_response(stream_query)
            if chunks:
                st.write_stream(chunks)
            else:
                st.error("Failed to stream the response.")

else:
    st.info("Please update settings and build/load the index in the sidebar before proceeding.")
