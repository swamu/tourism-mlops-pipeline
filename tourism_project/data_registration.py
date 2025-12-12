import os
import pandas as pd
from huggingface_hub import HfApi

def register_dataset():
    """Upload dataset to Hugging Face Hub"""
    
    # Get HF token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    # Initialize HF API
    api = HfApi()
    
    # Load and verify dataset
    dataset_path = "tourism_project/data/tourism.csv"
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    
    # Set repository ID
    repo_id = "swamu/tourism-dataset"
    
    try:
        # Upload the CSV file to HF Hub
        api.upload_file(
            path_or_fileobj=dataset_path,
            path_in_repo="tourism.csv",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token
        )
        print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    register_dataset()

