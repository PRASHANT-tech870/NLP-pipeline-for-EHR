import os
import gdown
import torch

def download_model_if_needed(file_path, gdrive_url):
    """
    Downloads the model file from Google Drive if it doesn't exist locally.
    
    Args:
        file_path (str): Local path where the model should be saved
        gdrive_url (str): Google Drive URL for the model file
    """
    if not os.path.exists(file_path):
        print(f"Downloading model to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        gdown.download(gdrive_url, file_path, quiet=False)
        print("Download complete!")
    return file_path 