import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/HarshVerma1233/MLOps-Mini-Project.mlflow')
dagshub.init(repo_owner='HarshVerma1233', repo_name='MLOps-Mini-Project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)