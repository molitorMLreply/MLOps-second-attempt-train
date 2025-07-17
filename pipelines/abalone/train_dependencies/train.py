import argparse
import sys
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    return pickle.load(model_path)

def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--model-type', type=str, required=True)
    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    return parser.parse_known_args()

def load_data(path):
    df = pd.read_csv(os.path.join(path, "train.csv"))
    y = df.iloc[:,0]
    X = df.iloc[:, 1:]
    return X, y

def main():
    args, _ = parse_args()
    X, y = load_data(args.train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    learning_rate = args.learning_rate
    model_type = args.model_type
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(learning_rate = learning_rate)
        model.fit(X = X_train, y = y_train)
        value = mean_squared_error(y_true = y_test, y_pred = model.predict(X_test))
    elif model_type == "randomforest":
        model = RandomForestRegressor()
        model.fit(X = X_train, y = y_train)
        value = mean_squared_error(y_true = y_test, y_pred = model.predict(X_test))
    elif model_type == "mlp":
        model = MLPRegressor(learning_rate_init=learning_rate)
        model.fit(X = X_train, y = y_train)
        value = mean_squared_error(y_true = y_test, y_pred = model.predict(X_test))
    elif model_type == "pytorch":
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        batch_size = args.batch_size
        dims = X_train.shape[1]
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        test_ds = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
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

        def train_model(model, train_loader, criterion, optimizer, epochs):
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for xb, yb in train_loader:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

        def evaluate_model(model, test_loader, criterion):
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    total_loss += loss.item()

        model = SimpleNet(dims = dims)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        n_epochs = args.epochs
        train_model(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, epochs = n_epochs)
        evaluate_model(model = model, test_loader = test_loader, criterion = criterion)
        predictions = model(X_test_tensor)
        value = torch.mean((y_test_tensor - predictions) ** 2 )
    print(f"validation:rmse={value}")
    if model_type == "pytorch":
        general_model_type = "pytorch"
    elif model_type == "xgboost":
        general_model_type = "xgboost"
    else:
        general_model_type = "sklearn"
    metadata = {"model_type": general_model_type}
    with open(os.path.join(args.model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f)
    if (general_model_type == "sklearn") or (general_model_type == "xgboost"):
        with open(os.path.join(args.model_dir, "model.pkl"), "wb") as file:
            pickle.dump(model, file)
    elif general_model_type == "pytorch":
        torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    else:
        raise TypeError("Model is not one of sklearn, xgboost, or pytorch")
    

if __name__ =='__main__':
    main()

