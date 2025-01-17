import streamlit as st
import pandas as pd
import json
from streamlit_lottie import st_lottie
from Utils.func import func
from Utils.rec_filter import rec_filter, fs
import os
from groq import Groq

# Page configuration
st.set_page_config(
    page_title="Medical NLP Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Get the current file's directory and load animation
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "Utils", "medical.json")
lottie_medical = load_lottiefile(json_path)

# Enhanced CSS with animations and modern styling
st.markdown("""
    <style>
    /* Main theme */
    .main {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling with gradient */
    h1 {
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 1rem;
        border-radius: 10px;
        background-color: #1e2433;
        color: white;
        border: 2px solid #2e3649;
        transition: all 0.3s ease;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 210, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Add this after your other imports
client = Groq(
    api_key="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"
)

def get_professional_analysis(df):
    """Get professional medical analysis from Groq"""
    try:
        # Convert DataFrame to a readable format
        results_text = df.to_string()
        
        prompt = f"""As a medical professional, analyze the following patient symptoms and provide a clear, professional summary. 
        Include any concerning patterns and explain the significance of the findings. Here's the data:

        {results_text}

        Please provide:
        1. A summary of key symptoms and their durations
        2. Any concerning combinations of symptoms
        3. Professional recommendations
        4. Whether further tests are needed (besides MRI if already recommended)

        Format the response in a clear, medical professional style."""

        # Get response from Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# App header with animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(lottie_medical, height=250, key="medical")
    st.markdown("<h1>Medical NLP Analyzer</h1>", unsafe_allow_html=True)

# Input section with description
st.markdown("""
    <div style='background-color: #1e2433; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h4 style='color: #00d2ff; margin-top: 0;'>📝 Enter Medical Text</h4>
        <p style='color: #ffffff99;'>Input your medical text below for comprehensive analysis. Our advanced NLP system will:</p>
        <ul style='color: #ffffff99; margin-left: 1.5rem;'>
            <li>Extract and identify symptoms</li>
            <li>Determine symptom durations</li>
            <li>Map associated organs</li>
            <li>Analyze severity to recommend MRI if needed</li>
        </ul>
        <p style='color: #ffffff99; font-style: italic; margin-top: 1rem;'>
            💡 The system uses state-of-the-art medical NLP to provide accurate analysis and MRI recommendations based on symptom patterns.
        </p>
        <div style='background: linear-gradient(135deg, #3a7bd520, #00d2ff10); border: 1px solid #00d2ff; 
             border-radius: 8px; padding: 1rem; margin-top: 1rem;'>
            <p style='color: #00d2ff; margin: 0;'>
                ⚠️ First-time users: The application needs to download required model weights (~400MB) on initial startup. 
                This may take a few minutes depending on your internet connection. Please be patient while the models are being loaded.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

medical_text = st.text_area(
    label="Medical Text Input",
    height=150,
    placeholder="Enter your medical text here...",
    label_visibility="collapsed"
)

# Submit button with loading state
if st.button("Analyze Text"):
    with st.spinner('Analyzing medical text...'):
        df = func(rec_filter(medical_text, fs))
        df1 = rec_filter(medical_text, fs)
        # Use consistent column names (uppercase)
        df1 = pd.DataFrame(df1, columns=['Symptom', 'Duration', 'Organ'])
        df2 = pd.DataFrame(df, columns=['Symptom', 'Result'])
        result = pd.merge(df1, df2, how='outer', left_on='Symptom', right_on='Symptom')
        df = result.fillna('')

    if not df.empty:
        # Results section with explanation
        st.markdown("""
            <div style='background-color: #1e2433; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;'>
                <h4 style='color: #00d2ff; margin-top: 0;'>🔍 Analysis Results</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Display the DataFrame
        st.dataframe(df, height=400)

        # Add explanation as an expander
        with st.expander("ℹ️ Understanding the Results"):
            st.markdown("""
                ### Duration Numbers:
                - 1: Hours
                - 2: Days
                - 3: Months
                - 4: Year
                - 5: seconds
                - 6: Minutes
                - 7: Years
                - 8: No duration specified

                ### Organ Field:
                - 'nil organ': No specific organ was found associated with this symptom
            """)

        # Check for MRI requirement in the Result column
        if "MRI NEEDED NOW !!!" in df['Result'].values:
            st.markdown("""
                <div class='mri-warning'>
                    <h4 style='color: #ff4b4b; margin-top: 0;'>⚠️ Urgent: MRI Analysis Required</h4>
                    <p>Critical symptoms detected. Please proceed to the 
                    <a href='http://localhost:7860/' target='_blank'>MRI image analysis page</a> 
                    immediately for further evaluation.</p>
                </div>
            """, unsafe_allow_html=True)

        # Add professional analysis
        st.markdown("""
            <div style='background-color: #1e2433; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;'>
                <h4 style='color: #00d2ff; margin-top: 0;'>🩺 Professional Analysis</h4>
            </div>
        """, unsafe_allow_html=True)
        
        with st.spinner('Generating professional analysis...'):
            analysis = get_professional_analysis(df)
            st.markdown(f"""
                <div style='background-color: #1e2433; padding: 1.5rem; border-radius: 10px;'>
                    {analysis}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No symptoms were detected in the provided text. Please try again with different text.")

# Footer
st.markdown("""
    <div style='position: fixed; bottom: 0; left: 0; right: 0; background-color: #1e2433; 
    padding: 1rem; text-align: center; font-size: 0.8rem; color: #ffffff99;'>
        Made with ❤️ at RVCE
    </div>
""", unsafe_allow_html=True)