export MLFLOW_FLASK_SERVER_SECRET_KEY="my-secret-key"
export MLFLOW_AUTH_CONFIG_PATH="basic_auth.ini"
mlflow server --backend-store-uri sqlite:///mlflow.db --app-name=basic-auth --host 0.0.0.0 --allowed-hosts '*' --port 5000