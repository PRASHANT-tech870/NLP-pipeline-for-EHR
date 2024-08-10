import gradio as gr
import requests
from PIL import Image
import base64
from io import BytesIO

url = "http://localhost:8002/WsEcl/json/query/roxie/mri_roxie22.1.1"

def predict_mri(uploaded_file):
    if uploaded_file is not None:
        print(f"Filepath: {uploaded_file}")  # Debug: Print the file path

        try:
            img = Image.open(uploaded_file)  # Open the image from the file path
            img.show()  # This should open the image for visual confirmation

            # Convert image to BytesIO object
            buffered = BytesIO()
            img.save(buffered, format="JPEG")  # Save the image in JPEG format to the buffer
            img_bytes = buffered.getvalue()  # Get the byte data

            # Encode the byte data to base64
            img_str = base64.b64encode(img_bytes).decode('utf-8')

            data = {
                "mri_roxie22.1.1": {
                    "img_string": img_str
                }
            }

            print("Sending data to server:", data)  # Debugging line

            try:
                response = requests.post(url, json=data, timeout=600)  # Set timeout to 600 seconds
                response.raise_for_status()  # Raise an error for bad responses
                json_response = response.json()
                predicted_label = json_response
                return f"Predicted label: {predicted_label}"
            except requests.exceptions.HTTPError as http_err:
                return f"HTTP error occurred: {http_err}"
            except requests.exceptions.Timeout:
                return "Error: Timeout occurred while processing the request."
            except Exception as e:
                return f"Error: {str(e)}"
        except Exception as e:
            return f"Error opening image: {str(e)}"
    else:
        return "Error: No image uploaded"

# Gradio interface
iface = gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="filepath"),  # Use 'filepath' to pass the file path to the function
    outputs="text",  # Outputs the prediction result as text
    title="MRI Classification",
    description="Upload an MRI image and click 'Submit' to get the classification result."
)

# Launch the interface
iface.launch()
