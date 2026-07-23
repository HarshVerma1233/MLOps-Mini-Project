import numpy as np
import pandas as pd
import os
import yaml
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import dagshub

# --- 1. SET CREDENTIALS FIRST ---
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    logging.warning("DAGSHUB_PAT environment variable is not set! Model logging will fall back to local storage.")
else:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "HarshVerma1233"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# --- 2. INITIALIZE DAGSHUB & MLFLOW TRACKING ---
repo_owner = 'HarshVerma1233'
repo_name = 'MLOps-Mini-Project'

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Unexpected error loading params: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        n_estimators = params.get('model_building', {}).get('n_estimators', 100)
        random_state = params.get('model_building', {}).get('random_state', 42)
        
        logger.debug("Loading processed feature datasets...")
        train_df = pd.read_csv(os.path.join('data/processed', 'train_features.csv'))
        test_df = pd.read_csv(os.path.join('data/processed', 'test_features.csv'))
        raw_train_df = pd.read_csv(os.path.join('data/interim', 'train.csv'))
        
        X_train = train_df.drop(columns=['label']).values
        y_train = train_df['label'].values
        
        X_test = test_df.drop(columns=['label']).values
        y_test = test_df['label'].values

        mlflow.set_experiment("Tweet_Emotion_Classification")
        
        with mlflow.start_run() as run:
            logger.info(f"Started remote MLflow run ID: {run.info.run_id}")
            logger.debug("Training model with n_estimators=%d...", n_estimators)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train)
            text_model = Pipeline(
                [
                    ("vectorizer", pickle.load(open("models/vectorizer.pkl", "rb"))),
                    ("classifier", RandomForestClassifier(
                        n_estimators=n_estimators, random_state=random_state
                    )),
                ]
            )
            text_model.fit(raw_train_df["content"].fillna(""), raw_train_df["sentiment"])
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            logger.debug("Train Accuracy: %.4f | Test Accuracy: %.4f", train_acc, test_acc)
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            
            logger.info("Uploading model artifacts to DagsHub MLflow server...")
            mlflow.sklearn.log_model(
                sk_model=text_model,
                name="model"
            )
            
            # Save local copy for DVC
            os.makedirs('models', exist_ok=True)
            model_path = os.path.join('models', 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            logger.debug('Model successfully built and saved locally to %s', model_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        raise e

if __name__ == '__main__':
    main()