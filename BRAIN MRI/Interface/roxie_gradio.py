import gradio as gr
import requests
from PIL import Image
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# Define URLs for each model endpoint
urls = {
    "Stroke Model": "http://localhost:8002/WsEcl/json/query/roxie/roxie_stroke_18.1",
    "Alzheimer Model": "http://localhost:8002/WsEcl/json/query/roxie/alzheimer_mri_roxie04.1",
    "Tumor Model": "http://localhost:8002/WsEcl/json/query/roxie/mri_roxie22.1.1"
}

def predict_mri(uploaded_file):
    if uploaded_file is not None:
        print(f"Filepath: {uploaded_file}")  # Debug: Print the file path

        try:
            # Open and process the image
            img = Image.open(uploaded_file)
            img.show()  # Optional: Opens the image for visual confirmation

            # Convert image to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode('utf-8')

            # Prepare data for each model
            data1 = {"roxie_stroke_18.1": {"img_string": img_str}}
            data2 = {"alzheimer_mri_roxie04.1": {"img_string": img_str}}
            data3 = {"mri_roxie22.1.1": {"img_string": img_str}}

            # Define separate request functions for each model
            def make_request1():
                try:
                    response = requests.post(urls["Stroke Model"], json=data1, timeout=600)
                    response.raise_for_status()
                    return "Stroke Model", response.json()
                except requests.exceptions.HTTPError as http_err:
                    return "Stroke Model", f"HTTP error occurred: {http_err}"
                except requests.exceptions.Timeout:
                    return "Stroke Model", "Error: Timeout occurred while processing the request."
                except Exception as e:
                    return "Stroke Model", f"Error: {str(e)}"
                
            def make_request2():
                try:
                    response = requests.post(urls["Alzheimer Model"], json=data2, timeout=600)
                    response.raise_for_status()
                    return "Alzheimer Model", response.json()
                except requests.exceptions.HTTPError as http_err:
                    return "Alzheimer Model", f"HTTP error occurred: {http_err}"
                except requests.exceptions.Timeout:
                    return "Alzheimer Model", "Error: Timeout occurred while processing the request."
                except Exception as e:
                    return "Alzheimer Model", f"Error: {str(e)}"
                
            def make_request3():
                try:
                    response = requests.post(urls["Tumor Model"], json=data3, timeout=600)
                    response.raise_for_status()
                    return "Tumor Model", response.json()
                except requests.exceptions.HTTPError as http_err:
                    return "Tumor Model", f"HTTP error occurred: {http_err}"
                except requests.exceptions.Timeout:
                    return "Tumor Model", "Error: Timeout occurred while processing the request."
                except Exception as e:
                    return "Tumor Model", f"Error: {str(e)}"

            # Execute requests in parallel
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(make_request1),
                    executor.submit(make_request2),
                    executor.submit(make_request3)
                ]
                results = {future.result()[0]: future.result()[1] for future in futures}

            # Format results as a table
            
            result_table = "| Model | Prediction |\n|-------|------------|\n"
            for model_name, prediction in results.items():
                result_table += f"| {model_name} | {prediction} |\n"

            return result_table
        except Exception as e:
            return f"Error opening image: {str(e)}"
    else:
        return "Error: No image uploaded"

# Gradio interface
iface = gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="filepath"),
    outputs="markdown",  # Output as markdown for table display
    title="MRI Classification",
    description="Upload an MRI image and click 'Submit' to get classification results from all models."
)

# Launch the interface
iface.launch()
