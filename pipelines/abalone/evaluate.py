"""Evaluation script for measuring mean squared error."""

import subprocess
import sys 

#subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.10.0", "xgboost==1.5.2"])


import json
import logging
import pathlib
import pickle
import tarfile
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0]
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = df

    model_path_init = "/opt/ml/processing/model/"
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    with open("model_metadata.json") as f:
        metadata = json.load(f)

    model_type = metadata["model_type"]
    logger.debug("Loading model.")
    if model_type == "sklearn":
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
    elif model_type == "xgboost":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.5.2"])
        import xgboost as xgb
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
    elif model_type == "pytorch":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.10.0"])
        import torch
        import torch.nn as nn
        class SimpleNet(nn.Module):
            def __init__(self, dims = 10):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(dims,16)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(16,dims)
                self.relu2 = nn.ReLU()
                self.out = nn.Linear(dims,1)
            def forward(self, x):
                return self.out(self.relu2(self.fc2(self.relu1(self.fc1(x)))))



        model = SimpleNet(dims = X_test.shape[1])
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    logger.info("Performing predictions against test data.")
    if model_type != "pytorch":
        predictions = model.predict(X_test)
        logger.debug("Calculating mean squared error.")
        mse = mean_squared_error(y_test, predictions)
        std = np.std(y_test - predictions)
    else:
        predictions = model(X_test)
        logger.debug("Calculating mean squared error.")
        mse = torch.mean((y_test - predictions) ** 2 )
        std = torch.std(y_test - predictions)
        mse = mse.item()
        std = std.item()
    
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
