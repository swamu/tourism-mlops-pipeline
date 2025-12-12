import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

def prepare_data():
    """Load, clean, split, and upload datasets"""
    
    # Get HF token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    repo_id = "swamu/tourism-dataset"
    
    print("Loading dataset from local file...")
    df = pd.read_csv("tourism_project/data/tourism.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Data cleaning - Remove unnecessary columns
    print("\nCleaning data...")
    columns_to_drop = ['Unnamed: 0', 'CustomerID']
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    # Split the dataset
    print("\nSplitting dataset...")
    train_df, test_df = train_test_split(
        df_cleaned,
        test_size=0.2,
        random_state=42,
        stratify=df_cleaned['ProdTaken']
    )
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    
    # Create directory and save locally
    os.makedirs("tourism_project/data/processed", exist_ok=True)
    train_df.to_csv("tourism_project/data/processed/train.csv", index=False)
    test_df.to_csv("tourism_project/data/processed/test.csv", index=False)
    print("\nDatasets saved locally")
    
    # Upload to HF Hub
    print("\nUploading to Hugging Face Hub...")
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj="tourism_project/data/processed/train.csv",
        path_in_repo="train.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )
    
    api.upload_file(
        path_or_fileobj="tourism_project/data/processed/test.csv",
        path_in_repo="test.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )
    
    print(f"\nData preparation complete! Datasets uploaded to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    prepare_data()
