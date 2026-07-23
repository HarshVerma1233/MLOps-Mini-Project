import numpy as np
import pandas as pd
import os
import json
import pickle
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- SET UP DAGSHUB CREDENTIALS ---
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    logging.warning("DAGSHUB_PAT environment variable is not set!")
else:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "HarshVerma1233"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner='HarshVerma1233', repo_name='MLOps-Mini-Project', mlflow=True)

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    try:
        # Load local model
        model_path = os.path.join('models', 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load test dataset
        test_df = pd.read_csv(os.path.join('data/processed', 'test_features.csv'))
        X_test = test_df.drop(columns=['label']).values
        y_test = test_df['label'].values

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        logger.info("Evaluation Metrics - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", acc, prec, rec, f1)

        # Retrieve the latest active run ID directly from DagsHub MLflow remote
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Tweet_Emotion_Classification")
        
        if experiment is None:
            raise ValueError("Experiment 'Tweet_Emotion_Classification' not found on MLflow remote.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1
        )

        if not runs:
            raise ValueError("No existing runs found in experiment 'Tweet_Emotion_Classification'.")

        latest_run_id = runs[0].info.run_id
        logger.info(f"Targeting active remote run ID for evaluation metrics: {latest_run_id}")

        # Log evaluation metrics to the retrieved remote run
        with mlflow.start_run(run_id=latest_run_id):
            mlflow.log_metric("eval_accuracy", acc)
            mlflow.log_metric("eval_precision", prec)
            mlflow.log_metric("eval_recall", rec)
            mlflow.log_metric("eval_f1_score", f1)

        # Save metrics locally
        os.makedirs('reports', exist_ok=True)
        metrics_path = os.path.join('reports', 'metrics.json')
        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Write experiment info containing the verified run ID for model_registration stage
        exp_info_path = os.path.join('reports', 'experiment_info.json')
        exp_info = {
            'run_id': latest_run_id,
            'experiment_name': "Tweet_Emotion_Classification"
        }
        with open(exp_info_path, 'w') as f:
            json.dump(exp_info, f, indent=4)

        logger.debug("Evaluation metrics and experiment info saved to reports/")

    except Exception as e:
        logger.error("Failed during model evaluation: %s", e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()