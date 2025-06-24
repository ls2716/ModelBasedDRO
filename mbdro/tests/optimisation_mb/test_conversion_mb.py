"""Test learning the conversoin model."""

import pytest
from mbdro.optimisation_mb.learn_conversion import train_model
from mbdro.envs.simple import SimpleEnv
import torch
import numpy as np


@pytest.fixture(scope="module")
def data():
    """Fixture to generate data."""
    np.random.seed(42)  # Set seed for reproducibility
    torch.manual_seed(42)
    env = SimpleEnv(alpha=-10, beta=0.5)
    X, y, r = env.generate_data(num_samples=500)
    return X, y


def test_train_model(data):
    """Test the training of the logistic regression model."""
    X, y = data

    # Train the model
    model, info = train_model(X, y, lr=0.1, no_iterations=1000, verbose=True)

    # Check the model parameters
    assert isinstance(model, torch.nn.Module), "Model should be a PyTorch module."
    assert len(list(model.parameters())) == 2, "Model should have two parameters."

    print("Model parameters:")
    for param in model.parameters():
        print(param)
    # Get the weight and bias
    alpha = model.alpha.item()
    beta = model.beta.item()
    print(f"Alpha: {alpha}, Beta: {beta}")

    # Assert that alpha and beta are close to the expected values
    assert np.isclose(alpha, -10., rtol=0.1), f"Alpha is not close to expected value: {alpha}"
    assert np.isclose(beta, 0.5, rtol=0.1), f"Beta is not close to expected value: {beta}"

    price = torch.tensor(10.0, dtype=torch.float32).reshape(-1, 1)
    acceptance_prob = model(price)

    profit = price * acceptance_prob
    assert profit.shape == (1, 1), "Profit shape is incorrect."
    assert acceptance_prob.shape == (1, 1), "Acceptance probability shape is incorrect."



def test_synthetic_data():
    """Test the training of the logistic regression model with synthetic data."""
    np.random.seed(42)  # Set seed for reproducibility
    torch.manual_seed(42)
    X = np.array([0.5, 0.5]).reshape(-1, 1)  # Single feature
    y = np.array([1, 0]).reshape(-1,1)  # Single sample


    # Train the model
    model, info = train_model(X, y, lr=0.1, no_iterations=1000, verbose=True)

    # Check the model parameters
    assert isinstance(model, torch.nn.Module), "Model should be a PyTorch module."
    assert len(list(model.parameters())) == 2, "Model should have two parameters."

    print("Model parameters:")
    for param in model.parameters():
        print(param)

    assert False, "Test not implemented yet."