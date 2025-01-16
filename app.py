import streamlit as st
import pandas as pd
import json
from streamlit_lottie import st_lottie
from Utils.func import func
from Utils.rec_filter import rec_filter, fs
import os

# Function to load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the JSON file in Utils directory
json_path = os.path.join(current_dir, "Utils", "medical.json")

# Load Lottie animations
lottie_medical = load_lottiefile(json_path)

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


# Create a button to submit the query
if st.button("Submit"):
    # Call the function with the input string
    df = func(rec_filter(medical_text, fs))

    import pandas as pd

    # Assuming FUNC_TEMP(med_data) and t1 are DataFrames
    # Replace these with your actual DataFrames


    df1 = rec_filter(medical_text, fs)
    df1 = pd.DataFrame(df1, columns=['symptom', 'duration', 'organ'])
    df2 = pd.DataFrame(df, columns=['symptom', 'result'])

    # Perform a full outer join on the 'symptom' column
    result = pd.merge(df1, df2, how='outer', left_on='symptom', right_on='symptom')
    df = result.fillna('')
    # Check if the DataFrame is not empty
    if not df.empty:
        # Display symptoms in a DataFrame
        st.write("### Symptoms and Associated Data")
        st.write(df)

        # Check for critical conditions based on data
        if "MRI NEEDED NOW !!!" in df.get("Symptom", []):
            st.markdown("""
                **MRI Analysis Required!**
                Please proceed to the [MRI image analysis page](http://localhost:7860/) for further steps.
            """, unsafe_allow_html=True)
    else:
        st.warning("No symptoms data found.")