# TO-DO:
# Load the train and test data from the data/processed folder
# Train the model on the train data
# Save the model metrics and parameters

import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from get_data import read_params
import mlflow
from urllib.parse import urlparse


def evaluate_metrics(actual, pred):
    """
    Evaluate the metrics
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaulate(config_path):
    """
    Load the train and test data from the data/processed folder, Train the model on the train data, and Save the model metrics and parameters
    """
    config = read_params(config_path)

    random_state = config["base"]["random_state"]
    target = config["base"]["target_col"]

    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    model_path = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    df_train = pd.read_csv(train_data_path, sep=",", encoding="utf-8")
    df_test = pd.read_csv(test_data_path, sep=",", encoding="utf-8")

    # Formulate the train dataset
    X_train = df_train.drop(columns=[target], axis=1)
    y_train = df_train[target]

    # Formulate the test dataset
    X_test = df_test.drop(columns=[target], axis=1)
    y_test = df_test[target]

    # MLFlow configuration
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        # Train and fit the model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Evaluate and print the metrics
        rmse, mae, r2 = evaluate_metrics(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Save the model
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.log_model(model, "model", registered_model_name=mlflow_config["registered_model_name"])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="../params.yaml")
    parsed_args = args.parse_args()
    train_and_evaulate(config_path=parsed_args.config)
