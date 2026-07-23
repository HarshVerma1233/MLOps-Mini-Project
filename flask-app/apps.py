import os
import mlflow
import mlflow.pyfunc
import dagshub
from flask import Flask, request, jsonify, render_template_string

# --- 1. SETUP DAGSHUB CREDENTIALS & TRACKING ---
dagshub_token = os.getenv("DAGSHUB_PAT")

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "HarshVerma1233"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

repo_owner = 'HarshVerma1233'
repo_name = 'MLOps-Mini-Project'

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

app = Flask(__name__)
model_name = "Tweet_Emotion_Classifier"

def load_registered_model(name):
    """
    Safely load model across Production, Staging, or latest version.
    """
    client = mlflow.tracking.MlflowClient()
    
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.RestException:
        raise RuntimeError(
            f"Model '{name}' is not registered on DagsHub yet. "
            "Please run 'dvc repro -f' first to complete model registration."
        )

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"Registered model '{name}' has no versions.")

    # 1. Look for Production
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if prod_versions:
        print(f"Loading Production model version: {prod_versions[0].version}")
        return mlflow.pyfunc.load_model(f"models:/{name}/Production")

    # 2. Fallback to Staging
    staging_versions = [v for v in versions if v.current_stage == "Staging"]
    if staging_versions:
        print(f"Loading Staging model version: {staging_versions[0].version}")
        return mlflow.pyfunc.load_model(f"models:/{name}/Staging")

    # 3. Fallback to latest numerical version
    latest_ver = sorted(versions, key=lambda v: int(v.version), reverse=True)[0].version
    print(f"Loading latest model version: {latest_ver}")
    return mlflow.pyfunc.load_model(f"models:/{name}/{latest_ver}")

# Load the model during app startup
model = load_registered_model(model_name)


# --- 2. DEFINE ROUTES ---

# ROOT ROUTE (Fixes the 404 error on GET /)
@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tweet Emotion Classifier</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; max-width: 600px; }
                textarea { width: 100%; height: 100px; margin-bottom: 15px; padding: 10px; }
                button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
                #result { margin-top: 20px; padding: 15px; background: #f4f4f4; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h2>Tweet Emotion Classifier API</h2>
            <p>API Status: <strong style="color: green;">Online</strong></p>
            <hr>
            <h3>Test Prediction</h3>
            <form id="emotionForm">
                <textarea id="tweetText" placeholder="Type a tweet here (e.g., 'I love learning MLOps!')..."></textarea><br>
                <button type="submit">Predict Emotion</button>
            </form>
            <div id="result" style="display: none;"></div>

            <script>
                document.getElementById('emotionForm').onsubmit = async (e) => {
                    e.preventDefault();
                    const text = document.getElementById('tweetText').value;
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    resultDiv.innerText = 'Analyzing...';

                    try {
                        const res = await fetch('/predict', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({text: text})
                        });
                        const data = await res.json();
                        resultDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } catch (err) {
                        resultDiv.innerText = 'Error making prediction request.';
                    }
                };
            </script>
        </body>
        </html>
    ''')


# PREDICT ROUTE (POST /predict)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide a JSON payload with a "text" key.'}), 400

        tweet_text = data['text']
        
        # Predict using the loaded MLflow model
        prediction = model.predict([tweet_text])
        
        # Convert prediction to standard Python data type
        pred_value = prediction[0] if hasattr(prediction, '__getitem__') else prediction

        return jsonify({
            'status': 'success',
            'input_text': tweet_text,
            'prediction': str(pred_value)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)