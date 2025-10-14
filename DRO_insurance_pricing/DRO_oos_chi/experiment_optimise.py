"""Defines a function to execute a single run of the analysis."""

import pandas as pd
import torch
import os
import numpy as np


from learn_conversion import train_model, get_conversion_model
from standard_optimisation import optimise_standard
from dro import optimise_robust




def train_logistic_model(data, seed, ratio):
    # Train a test and test model on the given data and save it.
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = data.sample(frac=ratio, random_state=seed)
    test_data = data.drop(train_data.index)

    # Try to load the model from logistic_models directory
    try:
        train_data_model = get_conversion_model(
            f"ratio_{ratio}/seed_{seed}/train_logistic_model_{seed}.pth",
            train_data.shape[1] - 3,  # Exclude cust_id, UnscaledPrice and Sale columns
        )
        print("Loaded existing model for train data")
        test_data_model = get_conversion_model(
            f"ratio_{ratio}/seed_{seed}/test_logistic_model_{seed}.pth",
            test_data.shape[1] - 3,  # Exclude cust_id, UnscaledPrice and Sale columns
        )
        print("Loaded existing model for test data")
    except FileNotFoundError:
        print("Training model on train data")
        train_data_model = train_model(train_data, epochs=5000)
        print("Training model on test data")
        test_data_model = train_model(test_data, epochs=5000)
        # Save both models to the models directory
        torch.save(
            train_data_model.state_dict(),
            f"ratio_{ratio}/seed_{seed}/train_logistic_model_{seed}.pth",
        )
        torch.save(
            test_data_model.state_dict(), f"ratio_{ratio}/seed_{seed}/test_logistic_model_{seed}.pth"
        )
    return train_data_model, test_data_model, train_data, test_data


def optimise_standard_prices_tocsv(data, seed, ratio, recreate_results=False):
    """Run price optimisation using standard methods."""
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("-----Optimising standard prices-----")
    # Check if the results already exist
    if not recreate_results and os.path.exists(f"ratio_{ratio}/seed_{seed}/prices_standard_test.csv"):
        print("Standard prices already optimised, skipping...")
        return

    # Get the train model
    train_data_model, test_data_model, train_data, test_data = train_logistic_model(data, seed, ratio)
    # Optimise the standard prices on the train data
    prices_standard_train_on_train = optimise_standard(
        train_data, train_data_model, min_acceptance_prob=0.0
    )
    prices_standard_train_on_test = optimise_standard(
        test_data, train_data_model, min_acceptance_prob=0.0
    )

    # Save the prices to a CSV file
    train_prices = prices_standard_train_on_train["Predicted_Price"]
    train_prices.to_csv(f"ratio_{ratio}/seed_{seed}/prices_standard_train.csv", index=False)
    test_prices = prices_standard_train_on_test["Predicted_Price"]
    test_prices.to_csv(f"ratio_{ratio}/seed_{seed}/prices_standard_test.csv", index=False)


def optimise_robust_prices_tocsv(data, seed, delta, ratio, recreate_results=False):
    """Run price optimisation using robust methods."""
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"-----Optimising robust prices for delta {delta}-----")
    # Check if the results already exist
    if not recreate_results and os.path.exists(
        f"ratio_{ratio}/seed_{seed}/prices_robust_delta_{delta}_test.csv"
    ):
        print(f"Robust prices for {delta} already optimised, skipping...")
        return

    # Get the train model
    train_data_model, test_data_model, train_data, test_data = train_logistic_model(data, seed, ratio)

    prices_robust_train_on_train = optimise_robust(train_data, train_data_model, delta)
    prices_robust_train_on_test = optimise_robust(test_data, train_data_model, delta)

    # If the prices are NaN, print a warning and return
    if prices_robust_train_on_train["Predicted_Price"].isnull().any():
        print(f"Warning: NaN values found in robust prices for delta {delta}.")
        return

    # Save the prices to a CSV file
    prices_robust_train_on_train["Predicted_Price"].to_csv(
        f"ratio_{ratio}/seed_{seed}/prices_robust_delta_{delta}_train.csv", index=False
    )
    prices_robust_train_on_test["Predicted_Price"].to_csv(
        f"ratio_{ratio}/seed_{seed}/prices_robust_delta_{delta}_test.csv", index=False
    )

def run_seed(seed, data, deltas, ratio, recreate_results=False):
    """Run the analysis for a given seed."""
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Create a directory for the seed if it doesn't exist
    if not os.path.exists(f"ratio_{ratio}/seed_{seed}"):
        os.makedirs(f"ratio_{ratio}/seed_{seed}")

    # Train the standard prices
    optimise_standard_prices_tocsv(data, seed, ratio, recreate_results)

    # Train the robust prices for each delta
    for delta in deltas:
        optimise_robust_prices_tocsv(data, seed, delta, ratio, recreate_results)


if __name__ == "__main__":
    recreate_results = False  # Set to True to recreate results forcibly
    # Load the data
    data = pd.read_csv("../data/atoti/scaled_data.csv")

    # Specify train data split ratio
    ratio = 0.1  # 50% of the data will be used for training, 50% for testing

    # Specify seeds
    seeds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 10 seeds for robustness

    # Specify the delta for robust optimisation
    deltas = [0.001, 0.002, 0.003, 0.005, 0.01, 0.05, 0.1]

    # Run the analysis for each seed
    for seed in seeds:
        print(f"Running analysis for seed {seed} ratio {ratio}...")
        run_seed(seed, data, deltas, ratio=ratio, recreate_results=recreate_results)
        print(f"Finished analysis for seed {seed} ratio {ratio}\n")

