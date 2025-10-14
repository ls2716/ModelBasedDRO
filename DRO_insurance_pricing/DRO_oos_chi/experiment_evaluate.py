"""Defines a function to execute a single run of the analysis."""

import pandas as pd
import os
import json
import torch

from standard_optimisation import evaluate_mean_profit
from dro import evaluate_robust_profit  # noqa: F401

from experiment_optimise import train_logistic_model

torch.manual_seed(0)


def evaluate_seed(seed, ratio, recreate_results=False):
    """Evaluate the prices for a given seed."""
    print(f"-----Evaluating prices for seed {seed}-----")
    # Load the evaluation dictionary from the path
    evaluation_dict_path = f"ratio_{ratio}/seed_{seed}/evaluation_dict.json"
    if not os.path.exists(evaluation_dict_path):
        evaluation_dict = {}
    else:
        with open(evaluation_dict_path, "r") as f:
            evaluation_dict = json.load(f)

    # Load the data
    data = pd.read_csv("../data/atoti/scaled_data.csv")

    # Train the logistic regression model on the data
    train_data_model, test_data_model, train_data, test_data = train_logistic_model(
        data, seed, ratio
    )

    prices_to_evaluate = [
        "standard",
        "robust_delta_0.001",
        "robust_delta_0.002",
        "robust_delta_0.003",
        "robust_delta_0.005",
        "robust_delta_0.01",
        "robust_delta_0.05",
        "robust_delta_0.1",
    ]

    # For each price type and each evaluation delta, evaluate the prices
    for price_to_evaluate in prices_to_evaluate:
        # Try to load the prices from the CSV file
        try:
            prices = pd.read_csv(f"ratio_{ratio}/seed_{seed}/prices_{price_to_evaluate}_test.csv")
            print(f"Loaded prices for {price_to_evaluate}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prices for {price_to_evaluate} not found. Please run the optimisation first."
            )
        if price_to_evaluate in evaluation_dict and not recreate_results:
            print(f"Prices for {price_to_evaluate} already evaluated, skipping...")
            # Skip evaluation if the prices have already been evaluated
            continue
        print(f"Evaluating prices for {price_to_evaluate}")
        # Initialise the dataset for evaluation
        test_data_with_prices = test_data.copy()
        test_data_with_prices["Predicted_Price"] = prices["Predicted_Price"].values

        profit = evaluate_mean_profit(test_data_with_prices, test_data_model)[0]
        # Save the profit in the evaluation dictionary
        evaluation_dict[price_to_evaluate]= profit

        # Save the evaluation dictionary to a JSON file
        with open(evaluation_dict_path, "w") as f:
            json.dump(evaluation_dict, f, indent=4)
        print(f"Evaluation for {price_to_evaluate} completed and saved.")


if __name__ == "__main__":
    recreate_results = False  # Set to True to recreate results forcibly
    # Load the data
    data = pd.read_csv("../data/atoti/scaled_data.csv")

    # Specify the train data split ratio
    ratio = 0.1

    # Set the seed
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # Evaluate the prices for each seed
    for seed in seeds:
        evaluate_seed(seed, ratio, recreate_results)
