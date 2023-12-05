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


def evaluate_metrics(actual, pred):
    """
    Evaluate the metrics
    """
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2

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

    # Train and fit the model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate and print the metrics
    mse, mae, r2 = evaluate_metrics(y_test, y_pred)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    # Save the model
    model_file_path = os.path.join(model_path, "model_1.pickle")
    joblib.dump(model, model_file_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    train_and_evaulate(config_path=parsed_args.config)
