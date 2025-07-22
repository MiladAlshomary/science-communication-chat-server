import os
import requests
import json

# Configuration
# Make sure to set your API key as an environment variable
API_KEY = os.environ.get("FIREWORKS_API_KEY")
ACCOUNT_ID = "nid2107-80f709" # Replace with your account ID

MODEL_ID = "ft-llama-journalist-full-model-new" # Replace with your desired model ID
MODEL_PATH = "/home/malshomary/Development/communicating-science-to-the-public-project/models/llama3-trained-journalist-on-deepseek-full/" # Path to your local model files

BASE_URL = "https://api.fireworks.ai/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def check_model_exists():
    """Check if the model already exists."""
    print(f"Checking if model {MODEL_ID} already exists...")
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/models/{MODEL_ID}"
    
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            print(f"Model {MODEL_ID} already exists.")
            return True
        elif response.status_code == 404:
            print(f"Model {MODEL_ID} does not exist.")
            return False
        else:
            print(f"Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking model existence: {e}")
        return False


def create_model():
    """Create a model object."""
    print("Creating model...")
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/models"
    payload = {
        "modelId": MODEL_ID,
        "model": {
            "kind": "CUSTOM_MODEL",
            "baseModelDetails": {
                "checkpointFormat": "HUGGINGFACE",
                "worldSize": 1,
            },
        },
    }
    
    try:
        response = requests.post(url, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        print("Model created successfully.")
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error creating model: {e}")
        print(f"Response content: {e.response.text}")
        raise


def get_upload_urls():
    """Get signed upload URLs for model files."""
    print("Getting upload URLs...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path '{MODEL_PATH}' does not exist!")
        return None
    
    # Get all files in the directory
    all_files = []
    for f in os.listdir(MODEL_PATH):
        file_path = os.path.join(MODEL_PATH, f)
        if os.path.isfile(file_path):
            all_files.append(f)
    
    filename_to_size = {
        f: os.path.getsize(os.path.join(MODEL_PATH, f))
        for f in all_files
    }
    
    print(f"Files to upload: {list(filename_to_size.keys())}")
    print(f"Total files: {len(filename_to_size)}")

    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/models/{MODEL_ID}:getUploadEndpoint"
    payload = {
        "filenameToSize": filename_to_size,
        "enableResumableUpload": False 
    }
    
    try:
        response = requests.post(url, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        print("Upload URLs received.")
        return response.json()["filenameToSignedUrls"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error getting upload URLs: {e}")
        print(f"Response content: {e.response.text}")
        raise


def upload_files(upload_urls):
    """Upload model files using signed URLs."""
    print("Uploading files...")
    
    for filename, url in upload_urls.items():
        print(f"  Uploading {filename}...")
        file_path = os.path.join(MODEL_PATH, filename)
        file_size = os.path.getsize(file_path)
        
        headers = {
            "Content-Type": "application/octet-stream",
            "x-goog-content-length-range": f"{file_size},{file_size}"
        }
        
        try:
            with open(file_path, "rb") as f:
                response = requests.put(url, data=f, headers=headers)
            response.raise_for_status()
            print(f"    Successfully uploaded {filename}")
        except requests.exceptions.HTTPError as e:
            print(f"    Error uploading {filename}: {e}")
            raise
    
    print("All files uploaded successfully.")


def validate_upload():
    """Validate the uploaded model."""
    print("Validating model upload...")
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/models/{MODEL_ID}:validateUpload"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        print("Model validation successful. Your model is ready to be deployed.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error validating upload: {e}")
        print(f"Response content: {e.response.text}")
        raise


def main():
    """Main function to orchestrate the model upload process."""
    try:
        if check_model_exists():
            print("Model already exists. Exiting.")
            return
        
        create_model()
        urls = get_upload_urls()
        if urls:
            upload_files(urls)
            validate_upload()
        else:
            print("No valid upload URLs received. Exiting.")
    except Exception as e:
        print(f"Error during model upload process: {e}")
        raise


if __name__ == "__main__":
    main()