import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize

from generate_data import generate_data



def train_model(demand_samples):
    """
    Train a model using the demand samples.
    
    Parameters:
    demand_samples (array-like): The demand samples to train the model on.
    
    Returns:
    tuple: The trained model parameters (mu, sigma).
    """
    # Define the negative log-likelihood function
    def nll(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf  # to ensure positive sigma
        a, b = (0 - mu) / sigma, np.inf
        logpdf = truncnorm.logpdf(demand_samples, a, b, loc=mu, scale=sigma)
        return -np.sum(logpdf)

    # Initial guess for the parameters
    initial_guess = [10, 10]

    # Optimize the parameters using MLE
    result = minimize(nll, initial_guess, bounds=[(-50, 100), (1e-6, None)])

    # Extract the optimized parameters
    mu_mle, sigma_mle = result.x

    return mu_mle, sigma_mle


if __name__ == "__main__":
    # Set random seed for reproducibility
    # seed = 0
    # np.random.seed(seed)

    # Set parameters for data generation
    num_samples = 100
    mean_demand = 20
    std_demand = 50
    # Generate demand samples
    demand_samples = generate_data(num_samples, mean_demand, std_demand)

    # Train the model
    mu_mle, sigma_mle = train_model(demand_samples)
    print(f"Trained model parameters: mu = {mu_mle}, sigma = {sigma_mle}")
    # Save the model parameters to a file

