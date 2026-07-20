# src/model/model_evaluation.py
import os
import json
import pickle
import logging
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_evaluation')

# Set up DagsHub credentials for MLflow tracking remote communication
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    logger.warning("DAGSHUB_PAT environment variable is not set. Fetching remote MLflow runs may fail.")
else:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "HarshVerma1233"
repo_name = "MLOps-Mini-Project"

# Point tracking URI directly to your DagsHub repository
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def main():
    try:
        logger.info("Loading processed test features...")
        test_data = pd.read_csv('data/processed/test_features.csv')
        
        # Adjust target column name to match your dataset labels
        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label']
        
        logger.info("Loading trained model from models/model.pkl...")
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        logger.info("Running predictions on test set...")
        predictions = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        logger.info(f"Metrics calculated - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # Ensure report directory exists
        os.makedirs('reports', exist_ok=True)
        
        # 1. Save local metrics report for DVC tracking
        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        with open('reports/metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        # 2. Extract active or last historical MLflow run metadata
        run_id = "local_fallback_run"
        try:
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
                logger.info(f"Using current active run ID: {run_id}")
            else:
                logger.info("No active run found contextually. Querying remote tracking server for the latest run...")
                client = mlflow.tracking.MlflowClient()
                
                # Match the exact experiment name declared in model_building.py
                try:
                    exp = client.get_experiment_by_name("Tweet_Emotion_Classification")
                    exp_id = exp.experiment_id if exp else "0"
                    logger.info(f"Connected to experiment: Tweet_Emotion_Classification (ID: {exp_id})")
                except Exception:
                    exp_id = "0"

                # Pull latest run generated during model building
                runs = client.search_runs(
                    experiment_ids=[exp_id], 
                    order_by=["attributes.start_time DESC"], 
                    max_results=1
                )
                if runs:
                    run_id = runs[0].info.run_id
                    logger.info(f"Successfully retrieved recent run ID: {run_id}")
                else:
                    logger.warning("No runs found in history on tracking server under Tweet_Emotion_Classification.")
        except Exception as mlflow_err:
            logger.error(f"Failed to query backend MLflow server for run details: {mlflow_err}")
        
        # Save exact metadata mapping configuration for model registration stage
        info_dict = {
            "run_id": run_id,
            "model_path": "model"
        }
        with open('reports/experiment_info.json', 'w') as f:
            json.dump(info_dict, f, indent=4)
            
        logger.info("Evaluation summaries saved successfully to reports/metrics.json and reports/experiment_info.json")
        
    except FileNotFoundError as e:
        logger.error(f"File tracking error: {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during evaluation: {e}")
        raise e

if __name__ == '__main__':
    main()