import os
import json
import logging
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# --- SET UP DAGSHUB CREDENTIALS ---
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    logging.warning("DAGSHUB_PAT environment variable is not set!")
else:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "HarshVerma1233"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

repo_owner = 'HarshVerma1233'
repo_name = 'MLOps-Mini-Project'

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    try:
        exp_info_path = os.path.join('reports', 'experiment_info.json')
        if not os.path.exists(exp_info_path):
            raise FileNotFoundError(f"Missing {exp_info_path}. Ensure model_evaluation has executed.")

        with open(exp_info_path, 'r') as f:
            exp_info = json.load(f)

        run_id = exp_info.get('run_id')
        if not run_id:
            raise ValueError("No valid run_id found in experiment_info.json.")

        logger.debug(f"Model info loaded from {exp_info_path}")

        client = MlflowClient()
        run = client.get_run(run_id)
        model_outputs = run.outputs.model_outputs if run.outputs else []
        if not model_outputs:
            raise ValueError(
                f"Run {run_id} has no logged model outputs. "
                "Run model_building.py successfully before registering the model."
            )
        if len(model_outputs) > 1:
            raise ValueError(
                f"Run {run_id} has multiple logged model outputs; "
                "experiment_info.json must identify one model."
            )

        # MLflow 3 stores models logged with `name` outside the run artifact
        # tree. Using the model ID avoids DagsHub's incomplete runs:/ lookup.
        model_id = model_outputs[0].model_id
        model_uri = f"models:/{model_id}"
        registered_model_name = "Tweet_Emotion_Classifier"

        logger.info(f"Targeting model URI: {model_uri}")

        # Register the logged model directly by ID.
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

        logger.info(f"Successfully registered model '{registered_model_name}' (Version {model_version.version})")

        # Transition model to Staging stage
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True
        )
        logger.info(f"Transitioned model version {model_version.version} to Stage 'Staging'.")

    except Exception as e:
        logger.error(f"Error during model registration: {e}")
        print(f"Error: {e}")
        raise e

if __name__ == '__main__':
    main()