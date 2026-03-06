export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=mlflowadmin123
mlflow models serve -m "models:/IhsgRegressor/2" --no-conda -p 5001 --no-conda
