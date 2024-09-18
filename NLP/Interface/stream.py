import streamlit as st
import requests
import pandas as pd
import json
from streamlit_lottie import st_lottie

# Function to load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Set the URL and authentication credentials
url = "http://4.152.243.71:8002/WsEcl/json/query/roxie/gcn_crf_final"

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
        response = requests.post(url, json=data)
        response.raise_for_status()  # Ensure we raise an error for bad responses
        return response
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

# Create a button to submit the query
if st.button("Submit"):
    # Create a JSON payload
    data = {
        "gcn_crf_final": {
            "med_data": medical_text
        }
    }

    response = make_request(url, data)
    
    # Check if the request was successful
    if response is not None:
        try:
            # Extract the JSON response
            json_response = response.json()
            print(json_response)
            print(type(json_response))
            rows = json_response['gcn_crf_finalResponse']['Results']['result_1']['Row']

            # Initialize lists to collect the parsed data
            symptom_to_organ_list = []
            symptom_with_duration_list = []
            result_list = []

            # Iterate through the rows and append the respective values to the lists
            for row in rows:
                symptom_to_organ = row.get("symptom_to_organ", "")
                symptom_with_duration = row.get("symptom_with_duration", "")
                result = row.get("result", "")

                symptom_to_organ_list.append(symptom_to_organ)
                symptom_with_duration_list.append(symptom_with_duration)
                result_list.append(result)

            # Create a Pandas DataFrame with the parsed data
            df = pd.DataFrame({
                "symptom_to_organ": symptom_to_organ_list,
                "symptom_with_duration": symptom_with_duration_list
            })

            # Display the Pandas table
            st.write(df)

            # Display the first cell of the "result" column separately
            if result_list:
                st.write(f"{result_list[0]}")
            else:
                st.write("No results available.")

        except (ValueError, KeyError) as e:
            st.error(f"Error processing the JSON response: {e}")
    else:
        st.error("Error: Request failed with no response")
