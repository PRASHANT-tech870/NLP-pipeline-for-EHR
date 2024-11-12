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
url = "http://48.214.28.80:8002/WsEcl/json/query/roxie/new_try_200"

# Load Lottie animations
lottie_medical = load_lottiefile("/home/ubuntu/Downloads/medical.json")

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
        font-size: 40px;
        white-space: nowrap;
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
        "new_try_200": {
            "med_data": medical_text
        }
    }

    response = make_request(url, data)
    
    # Check if the request was successful
    if response is not None:
        try:
            # Parse JSON response
            json_response = response.json()
            
            # Check if 'new_try_200Response' key exists
            if 'new_try_200Response' in json_response:
                response_data = json_response['new_try_200Response']
                
                # Process results if available
                if 'Results' in response_data and 'result_1' in response_data['Results']:
                    rows = response_data['Results']['result_1']['Row']

                    # Separate results for display
                    symptoms_data = []
                    result_messages = []

                    for row in rows:
                        if "symptom" in row and "organ" in row and "duration" in row:
                            symptoms_data.append({
                                "Symptom": row["symptom"],
                                "Organ": row["organ"],
                                "Duration": row["duration"]
                            })
                        if "result" in row:
                            result_messages.append(row["result"])

                    # Display symptoms in a DataFrame
                    if symptoms_data:
                        df = pd.DataFrame(symptoms_data)
                        st.write("### Symptoms and Associated Data")
                        st.write(df)

                    # Display result messages
                    if result_messages:
                        st.write("### Result Messages:")
                        for message in result_messages:
                            st.write(f"- {message}")
                            
                            # Check if any message requires MRI analysis
                            if message == "MRI NEEDED NOW !!!":
                                # Display a clickable link to the MRI analysis page
                                st.markdown("""
                                    **MRI Analysis Required!**
                                    Please proceed to the [MRI image analysis page](http://localhost:7860/) for further steps.
                                """, unsafe_allow_html=True)

                    else:
                        st.write("No critical result messages available.")
                    
                else:
                    st.error("Expected data structure not found.")
                    st.write("Response structure:", json_response)

            # If 'gcn_crf_finalResponse' key exists and contains 'Exception', display error
            elif 'gcn_crf_finalResponse' in json_response:
                response_data = json_response['gcn_crf_finalResponse']
                if 'Results' in response_data and 'Exception' in response_data['Results']:
                    st.error("Error in query execution:")
                    for error in response_data['Results']['Exception']:
                        st.write(f"Source: {error.get('Source')}")
                        st.write(f"Code: {error.get('Code')}")
                        st.write(f"Message: {error.get('Message')}")
                else:
                    st.error("Unexpected structure under 'gcn_crf_finalResponse'.")
                    st.write("Response structure:", json_response)

            else:
                st.error("Unknown response format.")
                st.write("Full response content:", json_response)  # For debugging

        except (ValueError, KeyError) as e:
            st.error(f"Error processing the JSON response: {e}")
            st.write("Full response content for debugging:", response.text)  # Display raw response
    else:
        st.error("Error: Request failed with no response.")
