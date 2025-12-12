# Tourism Package Prediction MLOps Pipeline

[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/features/actions)
[![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-yellow)](https://huggingface.co/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

## Project Overview

An end-to-end MLOps pipeline for predicting customer purchase behavior for the Wellness Tourism Package. This project implements automated data processing, model training with experiment tracking, and deployment using CI/CD best practices.

## Business Problem

"Visit with Us," a leading travel company, needs to efficiently identify customers likely to purchase the Wellness Tourism Package. This ML solution automates customer targeting, improving marketing efficiency and conversion rates.

## Architecture

```
Data Registration → Data Preparation → Model Training → Model Deployment
↓ ↓ ↓ ↓
Hugging Face Train/Test Split MLflow Tracking Streamlit App
Dataset Hub Feature Engineering Model Selection Docker Container
```

## Features

- **Automated Data Pipeline**: Registers and preprocesses datasets on Hugging Face Hub
- **ML Experimentation**: Trains 6 algorithms with hyperparameter tuning and MLflow tracking
- **Best Model**: Gradient Boosting with F1 Score of 0.8522
- **CI/CD**: GitHub Actions workflow for automated testing and deployment
- **Interactive UI**: Streamlit web app for real-time predictions
- **Containerized**: Docker-based deployment on Hugging Face Spaces

## Model Performance

| Model | F1 Score | Accuracy | Precision | Recall | ROC AUC |
|-------|----------|----------|-----------|--------|---------|
| **Gradient Boosting** | **0.8522** | **0.9479** | **0.9394** | **0.7799** | **0.9787** |
| Bagging | 0.8293 | 0.9407 | 0.9297 | 0.7484 | 0.9835 |
| Random Forest | 0.7636 | 0.9213 | 0.9052 | 0.6604 | 0.9752 |
| Decision Tree | 0.7103 | 0.8983 | 0.7863 | 0.6478 | 0.8116 |
| AdaBoost | 0.3850 | 0.8414 | 0.7593 | 0.2579 | 0.8272 |

## Technologies

- **ML/Data**: Python, pandas, scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Deployment**: Streamlit, Docker, Hugging Face Spaces
- **CI/CD**: GitHub Actions
- **Version Control**: Git, Hugging Face Hub

## Project Structure

```
tourism-mlops-pipeline/
├── .github/workflows/
│ └── pipeline.yml # CI/CD workflow
├── tourism_project/
│ ├── data/ # Dataset files
│ ├── model_building/ # Training scripts
│ ├── deployment/ # Deployment files
│ ├── data_registration.py
│ ├── data_preparation.py
│ └── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9+
- Hugging Face account and token
- GitHub account

### Installation

1. Clone the repository:
```bash
git clone https://github.com/swamu/tourism-mlops-pipeline.git
cd tourism-mlops-pipeline
```

2. Install dependencies:
```bash
pip install -r tourism_project/requirements.txt
```

3. Set environment variables:
```bash
export HF_TOKEN=your_huggingface_token
```

### Running Locally

1. **Data Preparation**:
```bash
python tourism_project/data_preparation.py
```

2. **Model Training**:
```bash
python tourism_project/model_building/train.py
```

3. **Run Streamlit App**:
```bash
cd tourism_project/deployment
streamlit run app.py
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Registers dataset on Hugging Face
2. Prepares and splits data
3. Trains models with MLflow tracking
4. Deploys the best model to Hugging Face Spaces

### Setup GitHub Actions

Getting the automated pipeline running is straightforward:

1. **Configure the token**: Navigate to repository Settings > Secrets and variables > Actions, then add a new secret named `HF_TOKEN` with the Hugging Face token value
2. **Push the code**: Once pushed to the main branch, the workflow kicks off automatically
3. **Watch it run**: The Actions tab shows real-time progress of each pipeline stage

## Deployment

- **Model**: [huggingface.co/swamu/tourism-prediction-model](https://huggingface.co/swamu/tourism-prediction-model)
- **Live App**: [huggingface.co/spaces/swamu/tourism-prediction-app](https://huggingface.co/spaces/swamu/tourism-prediction-app)

## MLflow Tracking

View experiment runs and metrics:
```bash
mlflow ui
```
Navigate to `http://localhost:5000`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Authors

- Swati Mukherjee

## Acknowledgments

- Hugging Face for hosting infrastructure
- MLflow for experiment tracking

---

Star this repository if find it helpful!
