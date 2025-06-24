"""Defines a function to execute a single run of the analysis."""
import pandas as pd
import torch
import os


from learn_conversion import train_model, get_conversion_model
from standard_optimisation import optimise_standard
from dro import optimise_robust

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


def optimise_standard_prices_tocsv(data, seed, recreate_results=False):
    """Run price optimisation using standard methods."""
    print("-----Optimising standard prices-----")
    # Check if the results already exist
    if not recreate_results and os.path.exists(f"seed_{seed}/prices_standard.csv"):
        print("Standard prices already optimised, skipping...")
        return

    # Get the train model
    train_data_model = train_logistic_model(data, seed)
    prices_standard_train_on_train = optimise_standard(
        data, train_data_model, min_acceptance_prob=0.0
    )

    # Save the prices to a CSV file
    prices = prices_standard_train_on_train["Predicted_Price"]
    prices.to_csv(f"seed_{seed}/prices_standard.csv", index=False)


def optimise_robust_prices_tocsv(data, seed, delta, recreate_results=False):
    """Run price optimisation using robust methods."""
    print(f"-----Optimising robust prices for delta {delta}-----")
    # Check if the results already exist
    if not recreate_results and os.path.exists(
        f"seed_{seed}/prices_robust_delta_{delta}.csv"
    ):
        print(f"Robust prices for {delta} already optimised, skipping...")
        return

    # Get the train model
    train_data_model = train_logistic_model(data, seed)
    prices_robust_train_on_train = optimise_robust(data, train_data_model, delta)

    # If the prices are NaN, print a warning and return
    if prices_robust_train_on_train["Predicted_Price"].isnull().any():
        print(f"Warning: NaN values found in robust prices for delta {delta}.")
        return

    # Save the prices to a CSV file
    prices = prices_robust_train_on_train["Predicted_Price"]
    prices.to_csv(f"seed_{seed}/prices_robust_delta_{delta}.csv", index=False)


if __name__ == "__main__":
    recreate_results = False  # Set to True to recreate results forcibly
    # Load the data
    data = pd.read_csv("../data/atoti/scaled_data.csv")

    # Set the seed
    seed = 0

    # Specify the delta for robust optimisation
    deltas = [0.01, 0.05, 0.1]

    # Create a directory for the seed if it doesn't exist
    if not os.path.exists(f"seed_{seed}"):
        os.makedirs(f"seed_{seed}")

    # Train the standard prices
    optimise_standard_prices_tocsv(data, seed, recreate_results)

    # Train the robust prices for each delta
    for delta in deltas:
        optimise_robust_prices_tocsv(data, seed, delta, recreate_results)
