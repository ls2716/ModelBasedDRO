"""Learn the probabilities using logistic regression with MSE loss"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from generate_data import generate_data

def train_model(X, actions, labels):
    """
    Learn the conversion probabilities using logistic regression with MSE loss.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        actions (np.ndarray): Action vector of shape (n_samples,).
        labels (np.ndarray): Label vector of shape (n_samples,).

    Returns:
        np.ndarray: Learned conversion probabilities for each action.
    """

    # Concatenate features and one-hot actions
    X_augmented = np.hstack([X, actions])  # shape: (n_samples, n_features + n_actions)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_augmented, labels, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict probabilities on the test set
    probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    # Compute MSE loss
    mse_loss = mean_squared_error(y_test, probs)
    
    # print(f"MSE Loss: {mse_loss}")


    probs_augmented = model.predict_proba(X_augmented)[:, 1]  # Get probabilities for the positive class
    probs = probs_augmented.reshape(-1, 1)  # Reshape to (n_samples, 1)

    # Compute the mean probabilities corresponding to each action
    mean_probs = []
    for i in range(actions.shape[1]):
        mean_prob = np.mean(probs[actions[:, i] == 1])
        mean_probs.append(mean_prob)

    return model, np.array(mean_probs).flatten()



if __name__ == "__main__":
    seed = 10
    
    no_seeds_arr = [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 5000000]
    n_runs = len(no_seeds_arr)

    mean_probs_data = np.zeros((n_runs,5))
    mean_probs_model = np.zeros((n_runs,5))
    for i, n_samples in enumerate(no_seeds_arr):
        # Generate random data
        X, A_one_hot, labels, gen_dict = generate_data(parameter_seed=seed, n_samples=n_samples, verbose=False)

        mean_probs_data[i, :] = gen_dict["mean_probs"]

        # Learn conversion probabilities
        model, mean_probs = train_model(X, A_one_hot, labels)

        mean_probs_model[i, :] = mean_probs

    print("Mean probabilities from data:")
    print(mean_probs_data)
    print("Mean probabilities from model:")
    print(mean_probs_model)

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.semilogx(no_seeds_arr, mean_probs_data[:, i], label=f"Data Action {i}")
        plt.semilogx(no_seeds_arr, mean_probs_model[:, i], label=f"Model Action {i}", linestyle='--')
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Probability")
    plt.title("Mean Probabilities from Data vs Model")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot aboslute errors with respecto to the last sample
    true_mean_probs = mean_probs_data[-1, :]
    # Exclude the last sample from the mean_probs_data and mean_probs_model
    mean_probs_data = mean_probs_data[:-1, :]
    mean_probs_model = mean_probs_model[:-1, :]
    abs_errors = np.abs(mean_probs_data - true_mean_probs)
    abs_errors_model = np.abs(mean_probs_model - true_mean_probs)
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.loglog(no_seeds_arr[:-1], abs_errors[:, i], label=f"Data Action {i}")
        plt.loglog(no_seeds_arr[:-1], abs_errors_model[:, i], label=f"Model Action {i}", linestyle='--')
    plt.xlabel("Number of Samples")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error of Mean Probabilities from Data vs Model")
    plt.legend()
    plt.grid()
    plt.show()
    # Save the results

