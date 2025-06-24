"""Defines a function to execute a single run of the analysis."""

import pandas as pd
import torch
import os
import json


from learn_conversion import train_model, get_conversion_model
from standard_optimisation import evaluate_mean_profit
from dro import evaluate_robust_profit

torch.manual_seed(0)


def train_logistic_model(data, seed):
    """Train a logistic regression model on the given data and save it."""
    train_data = data
    # Train the conversion models for both the train and test data
    # Try to load the model from logistic_models directory
    try:
        train_data_model = get_conversion_model(
            f"seed_{seed}/all_data_logistic_model_{seed}.pth",
            train_data.shape[1] - 3,  # Exclude cust_id, UnscaledPrice and Sale columns
        )
        print("Loaded existing model for train data")
    except FileNotFoundError:
        print("Training model on train data")
        train_data_model = train_model(train_data, epochs=5000)

    # Save both models to the models directory
    torch.save(
        train_data_model.state_dict(), f"seed_{seed}/all_data_logistic_model_{seed}.pth"
    )
    return train_data_model


if __name__ == "__main__":
    recreate_results = False # Set to True to recreate results forcibly
    # Load the data
    data = pd.read_csv("../data/atoti/scaled_data.csv")

    # Set the seed
    seed = 0

    # Load the evaluation dictionary from the path
    evaluation_dict_path = f"seed_{seed}/evaluation_dict.json"
    if not os.path.exists(evaluation_dict_path):
        evaluation_dict = {}
    else:
        with open(evaluation_dict_path, "r") as f:
            evaluation_dict = json.load(f)

    prices_to_evaluate = [
        "standard",
        "robust_delta_0.01",
        "robust_delta_0.05",
        "robust_delta_0.1",
    ]

    # Train the logistic regression model on the data
    acceptance_model = train_logistic_model(data, seed)

    # Specify the delta for robust optimisation (0 for standard evaluation)
    deltas = [0, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1]
    # For each price type and each evaluation delta, evaluate the prices
    for price_to_evaluate in prices_to_evaluate:
        # Try to load the prices from the CSV file
        try:
            prices = pd.read_csv(f"seed_{seed}/prices_{price_to_evaluate}.csv")
            print(f"Loaded prices for {price_to_evaluate}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prices for {price_to_evaluate} not found. Please run the optimisation first."
            )
        if price_to_evaluate not in evaluation_dict:
            evaluation_dict[price_to_evaluate] = {}
        # Initialise the dataset for evaluation
        data_with_prices = data.copy()
        data_with_prices["Predicted_Price"] = prices["Predicted_Price"].values
        for delta in deltas:
            # Check if the evaluation for this price type and delta already exists
            if (
                not recreate_results
                and price_to_evaluate in evaluation_dict
                and str(delta) in evaluation_dict[price_to_evaluate]
            ):
                print(
                    f"Evaluation for {price_to_evaluate} with delta {delta} already exists, skipping..."
                )
                continue
            if delta == 0:
                print(
                    f"Evaluating mean profit for {price_to_evaluate} with standard evaluation."
                )
                profit = evaluate_mean_profit(data_with_prices, acceptance_model)[0]
            else:
                print(
                    f"Evaluating robust profit for {price_to_evaluate} with delta {delta}."
                )
                profit = evaluate_robust_profit(
                    data_with_prices, acceptance_model, delta
                )
            # Save the profit in the evaluation dictionary
            evaluation_dict[price_to_evaluate][str(delta)] = profit

            # Save the evaluation dictionary to a JSON file
            with open(evaluation_dict_path, "w") as f:
                json.dump(evaluation_dict, f, indent=4)
            print(f"Evaluation for {price_to_evaluate} completed and saved.")
