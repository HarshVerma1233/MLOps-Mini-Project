# src/model/register_model.py
import os
import json
import logging
import pickle
import mlflow
from mlflow.tracking import MlflowClient

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    try:
        # 1. Load the performance metrics to verify what we are registering
        metrics_path = os.path.join('reports', 'metrics.json')
        logger.debug("Reading evaluation metrics from %s...", metrics_path)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        logger.info("Registering model with Accuracy: %.4f, F1-Score: %.4f", 
                    metrics.get('accuracy', 0.0), metrics.get('f1_score', 0.0))

        # 2. Point to the MLflow experiment
        mlflow.set_experiment("Tweet_Emotion_Classification")
        
        # Start a run to log and register the model artifact
        with mlflow.start_run() as run:
            # Log metrics into this registration run context as well
            mlflow.log_metrics({
                "reg_accuracy": metrics.get('accuracy', 0.0),
                "reg_f1": metrics.get('f1_score', 0.0)
            })
            
            # Load the local model object to log it into MLflow
            model_path = os.path.join('models', 'model.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Log the model artifact to MLflow runs
            logger.debug("Logging model artifact directly to MLflow...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="Tweet_Emotion_Classifier_Model"
            )
            
        logger.info("Model registered successfully in the MLflow Model Registry!")

    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()