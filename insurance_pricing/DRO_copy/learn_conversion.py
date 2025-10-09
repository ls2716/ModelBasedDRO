"""Scripts which contain learning conversion functions for given input data."""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add parent directory to path

from networks import LogisticRegression


# Set the random seed for reproducibility
torch.manual_seed(0)

def train_conversion_models(train_data, test_data, epochs):
    """Train the conversion models for both the train and test data."""
    train_data_model = train_model(train_data, epochs)
    test_data_model = train_model(test_data, epochs)
    return train_data_model, test_data_model

def train_model(data, epochs=10000):
    """"Train a logistic regression model on the given data."""
    # Drop the customer id column
    d = data.drop("cust_id", axis=1)
    # Drop the unscaled price column
    d = d.drop("UnscaledPrice", axis=1)

    # Split the data into features and target
    X = d.drop("Sale", axis=1)
    y = d["Sale"]

    # Convert the data to PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).clip(0, 1)  # Ensure target is binary

    # Instantiate the model
    model = LogisticRegression(shape=X.shape[1])

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train the model
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y.view(-1, 1))  # Ensure target shape matches output

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model


def evaluate_model(model, data, threshold=0.5):
    """Evaluate the model on the given data."""
    # Drop the customer id column
    d = data.drop("cust_id", axis=1)
    # Drop the unscaled price column
    d = d.drop("UnscaledPrice", axis=1)

    # Split the data into features and target
    X = d.drop("Sale", axis=1)
    y = d["Sale"]

    # Convert the data to PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).clip(0, 1)  # Ensure target is binary

    # Evaluate the model
    with torch.no_grad():
        y_pred = model(X)
        loss = nn.BCELoss()(y_pred, y.view(-1, 1))

    print("-------------------")
    print(f"Evaluation Loss: {loss.item():.3f}")
    # Check the model's accuracy, precision and f1 score
    y_pred_classes = (y_pred > threshold).float()
    accuracy = (y_pred_classes == y.view(-1, 1)).float().mean()
    print(f"Accuracy: {accuracy.item():.3f}")
    precision = (y_pred_classes * y.view(-1, 1)).sum() / (y_pred_classes.sum() + 1e-8)
    print(f"Precision: {precision.item():.3f}")
    f1 = 2 * (precision * accuracy) / (precision + accuracy + 1e-8)
    print(f"F1 Score: {f1.item():.3f}")
    print("-------------------")


def get_conversion_model(path, no_features):
    """Load the conversion model from the given path."""
    model = LogisticRegression(shape=no_features)
    model.load_state_dict(torch.load(path))
    return model


if __name__=="__main__":
    # Load the data
    data = pd.read_csv("../data/atoti/scaled_data.csv")

    seed = 1

    # Set the seed for data split
    train_data = data.sample(frac=0.8, random_state=seed)
    test_data = data.drop(train_data.index)

    # Train the conversion models
    train_data_model = train_model(train_data, epochs=8000)
    test_data_model = train_model(test_data, epochs=8000)

    # Evaluate the models on both train and test data
    print("Evaluating models")
    print("Evaluating train model on the train data")
    evaluate_model(train_data_model, train_data)
    print("Evaluating the train model on the test data")
    evaluate_model(train_data_model, test_data)
    print("Evaluating test model on the train data")
    evaluate_model(test_data_model, train_data)
    print("Evaluating the test model on the test data")
    evaluate_model(test_data_model, test_data)