import streamlit as st
import requests
import pandas as pd
import time
from streamlit_lottie import st_lottie
import json

# Function to load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Set the URL and authentication credentials
url = "http://52.167.25.124:8002/WsEcl/json/query/roxie/lets_try"



# Load Lottie animations
lottie_medical = load_lottiefile("medical.json")


# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: rgb(14, 17, 23);
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 20px;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1c6ea4;
    }
    .stTextInput>div>div>input {
        font-size: 20px;
        padding: 10px;
        border-radius: 8px;
        background-color: #333333;
        color: white;
    }
    .css-1q8dd3e, .css-1d391kg {
        background-color: #000000;
    }
    h1 {
        font-size: 40px;  /* Adjust the size as needed */
        white-space: nowrap; /* Prevents line break */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔍 Medical Entities Extractor using NLP</h1>", unsafe_allow_html=True)

st_lottie(lottie_medical, height=200, key="medical")

# Create a text input field
medical_text = st.text_input("Enter medical text:")
def make_request(url, data):
    try:
        response = requests.post(url, json=data, timeout=20)  # Set timeout to 15 seconds
        return response
    except requests.Timeout:
        return None

# Create a button to submit the query
if st.button("Submit"):
    # Create a JSON payload
    data = {
        "lets_try": {
            "med_data": medical_text
        }
    }

    response = make_request(url, data)
    retried = False  # Flag to track if a retry was needed
    
    # Check if the request timed out
    if response is None:
        retried = True
        st.warning("No response received within 15 seconds. Retrying...")
        time.sleep(1)  # Wait for 15 seconds
        response = make_request(url, data)  # Retry the request

    # Clear the warning if response is received after retry
    if response is not None and response.status_code == 200:
        # If the request was successful, extract the JSON response
        st.write("Response Recieved:")
        json_response = response.json()
        rows = json_response['lets_tryResponse']['Results']['result_1']['Row']

        # Create a Pandas DataFrame from the 'Row' data
        df = pd.DataFrame(rows)

        # Display the Pandas table
        st.write(df)
        
        # Remove warning if it was displayed
        if retried:
            st.empty()  # Clear the warning message
    else:
        # If the request failed, display an error message
        st.error("Error: Request failed with status code {}".format(response.status_code) if response else "Error: Request timed out again")