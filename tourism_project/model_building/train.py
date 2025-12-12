import os
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                             AdaBoostClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, create_repo
import shutil

def train_models():
    """Train multiple models, track with MLflow, and register best to HF"""
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv("tourism_project/data/processed/train.csv")
    test_df = pd.read_csv("tourism_project/data/processed/test.csv")
    
    # Preprocess
    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle missing values
    X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train[numerical_cols].median())
    X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train[numerical_cols].median())
    X_train[categorical_cols] = X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
    X_test[categorical_cols] = X_test[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
    
    # Encode and scale
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Setup MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Tourism_Package_Prediction")
    
    # Define models
    models_params = {
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [5, 7]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20]
            }
        }
    }
    
    # Train models
    results = []
    for model_name, mp in models_params.items():
        print(f"\nTraining {model_name}...")
        with mlflow.start_run(run_name=model_name):
            grid_search = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            mlflow.log_params(grid_search.best_params_)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.sklearn.log_model(best_model, "model")
            
            results.append({'Model': model_name, 'F1': metrics['f1_score'],
                          'Params': grid_search.best_params_, **metrics})
            print(f"{model_name} F1 Score: {metrics['f1_score']:.4f}")
    
    # Get best model
    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    print(f"\nBest Model: {best_model_name}")
    
    # Save artifacts
    os.makedirs("tourism_project/model_building", exist_ok=True)
    best_model_config = models_params[best_model_name]
    best_params = results_df.iloc[0]['Params']
    final_model = best_model_config['model'].set_params(**best_params)
    final_model.fit(X_train, y_train)
    
    artifacts = {
        'model': final_model, 'scaler': scaler, 'label_encoders': label_encoders,
        'feature_names': X_train.columns.tolist(), 'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    
    with open('tourism_project/model_building/model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    # Register to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        model_dir = "tourism_model"
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy('tourism_project/model_building/model_artifacts.pkl', f'{model_dir}/model_artifacts.pkl')
        
        with open(f'{model_dir}/README.md', 'w') as f:
            f.write(f"# Tourism Prediction Model\n\nBest Model: {best_model_name}\nF1 Score: {results_df.iloc[0]['F1']:.4f}")
        
        api = HfApi()
        repo_id = "swamu/tourism-prediction-model"
        create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
        api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model", token=hf_token)
        print(f"Model uploaded to https://huggingface.co/{repo_id}")
        shutil.rmtree(model_dir)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    train_models()
