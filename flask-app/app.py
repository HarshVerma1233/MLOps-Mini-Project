import os
from pathlib import Path

import dagshub
import mlflow
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "HarshVerma1233")
REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "MLOps-Mini-Project")
MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/Tweet_Emotion_Classifier/Staging",
)


def create_app(model=None):
    app = Flask(__name__)
    loaded_model = model

    def get_model():
        nonlocal loaded_model
        if loaded_model is None:
            token = os.getenv("DAGSHUB_PAT")
            if not token:
                raise RuntimeError("DAGSHUB_PAT is required to load the model from DagsHub.")
            os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
            mlflow.set_tracking_uri(
                f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
            )
            loaded_model = mlflow.pyfunc.load_model(MODEL_URI)
        return loaded_model

    @app.get("/")
    def home():
        return jsonify(
            {
                "service": "Tweet emotion classifier",
                "endpoint": "POST /predict",
                "model_uri": MODEL_URI,
            }
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            return jsonify({"error": "Request JSON must contain a string field named 'text'."}), 400

        text = payload["text"].strip()
        if not text:
            return jsonify({"error": "The 'text' field must not be empty."}), 400

        prediction = get_model().predict([text])
        label = int(prediction[0])
        return jsonify({"prediction": label, "emotion": "happy" if label == 1 else "sad"})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)