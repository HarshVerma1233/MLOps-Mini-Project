# src/model/model_evaluation.py
import numpy as np
import pandas as pd
import os
import yaml
import logging
import pickle
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/HarshVerma1233/MLOps-Mini-Project.mlflow')
dagshub.init(repo_owner='HarshVerma1233', repo_name='MLOps-Mini-Project', mlflow=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    try:
        # 1. Load the processed evaluation test data
        logger.debug("Loading processed test features...")
        test_df = pd.read_csv(os.path.join('data/processed', 'test_features.csv'))
        
        # Split features and labels
        X_test = test_df.drop(columns=['label']).values
        y_test = test_df['label'].values

        # 2. Load the trained model artifact built in the previous stage
        model_path = os.path.join('models', 'model.pkl')
        logger.debug("Loading trained model from %s...", model_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 3. Generate Predictions and Compute Performance Metrics
        logger.debug("Running predictions on test set...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        logger.debug("Metrics calculated - Accuracy: %.4f, F1-Score: %.4f", accuracy, f1)

        # 4. Log the Metrics to MLflow (attaches to the ongoing experiment)
        mlflow.set_experiment("Tweet_Emotion_Classification")
        
        # If running inside a DVC pipeline, we let it log standalone or append metrics
        with mlflow.start_run(nested=True):
            mlflow.log_metric("eval_accuracy", accuracy)
            mlflow.log_metric("eval_precision", precision)
            mlflow.log_metric("eval_recall", recall)
            mlflow.log_metric("eval_f1", f1)
            logger.debug("Logged metrics successfully to MLflow.")

        # 5. Build and save the json reports exactly where dvc.yaml expects them
        os.makedirs('reports', exist_ok=True)
        
        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        # Write to metrics.json
        metrics_path = os.path.join('reports', 'metrics.json') 
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        # Write to experiment_info.json
        info_path = os.path.join('reports', 'experiment_info.json') 
        with open(info_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        logger.debug("Evaluation summaries saved successfully to reports/metrics.json and reports/experiment_info.json")

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()