import os
from huggingface_hub import HfApi, create_repo
import shutil

def deploy_to_huggingface():
    """Deploy application to Hugging Face Spaces"""

# Get HF token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
raise ValueError("HF_TOKEN environment variable not set")

# Setup
api = HfApi()
space_id = "swamu/tourism-prediction-app" # UPDATE THIS with the username

print(" Preparing deployment files...")

# Create temporary deployment directory
deploy_dir = "temp_deploy"
if os.path.exists(deploy_dir):
shutil.rmtree(deploy_dir)
os.makedirs(deploy_dir)

# Copy deployment files
shutil.copy("tourism_project/deployment/app.py", f"{deploy_dir}/app.py")
shutil.copy("tourism_project/deployment/requirements.txt", f"{deploy_dir}/requirements.txt")
shutil.copy("tourism_project/deployment/Dockerfile", f"{deploy_dir}/Dockerfile")

# Create README for Space
readme_content = """---
title: Tourism Package Prediction
emoji:
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Wellness Tourism Package Prediction App

This application predicts whether a customer will purchase the Wellness Tourism Package based on their profile and interaction data.

## Features
- Real-time predictions using trained ML model
- Interactive web interface built with Streamlit
- Model loaded from Hugging Face Model Hub

## Usage
Enter customer information and get instant predictions on purchase likelihood.
"""

with open(f"{deploy_dir}/README.md", "w") as f:
    f.write(readme_content)

print(" Deploying to Hugging Face Spaces...")

try:
# Create Space
create_repo(
space_id,
repo_type="space",
space_sdk="docker",
exist_ok=True,
token=hf_token
)

# Upload all files
api.upload_folder(
folder_path=deploy_dir,
repo_id=space_id,
repo_type="space",
token=hf_token
)

print(f"\n Deployment successful!")
print(f" App URL: https://huggingface.co/spaces/{space_id}")
print("\n Note: It may take a few minutes for the Space to build and start.")

except Exception as e:
    print(f" Error during deployment: {e}")
raise
finally:
# Cleanup
if os.path.exists(deploy_dir):
shutil.rmtree(deploy_dir)
print("\n Cleanup complete!")

if __name__ == "__main__":
deploy_to_huggingface()
