"""Test Bayesian Optimisation"""

import pytest
import numpy as np
import torch

from mbdro.optimisation_bayesian.learn_conversion import train_model
from mbdro.optimisation_bayesian.dro_xi_bayesian import evaluate_robust_profit, bayesian_optimise_robust_price
from mbdro.optimisation_bayesian.standard_expectation import evaluate_profit, bayesian_optimise_price
from mbdro.optimisation_bayesian.standard_mle import evaluate_mle_profit, bayesian_optimise_mle_price
from mbdro.envs.simple import SimpleEnv


@pytest.fixture(scope="module")
def data():
    """Fixture to generate data."""
    # Set the seed
    np.random.seed(0)
    torch.manual_seed(0)
    env = SimpleEnv(alpha=-10, beta=0.5)
    X, y, r = env.generate_data(num_samples=800)

    prices = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
    acceptance_probabilities = env.compute_acceptance_probability(prices)
    rewards = prices * acceptance_probabilities
    # Return the optimal price and the corresponding reward
    best_price_index = np.argmax(rewards)
    best_price = prices[best_price_index, 0]
    best_reward = rewards[best_price_index, 0]
    return X, y, best_price, best_reward


@pytest.fixture(scope="module")
def acceptance_model(data):
    """Fixture to train the acceptance model."""
    X, y, _, _ = data
    # Train the acceptance model using the training data
    acceptance_model, info = train_model(X, y, num_samples=100)
    return acceptance_model


def test_evaluate_profit(acceptance_model):
    """Test the profit evaluation function."""
    env = SimpleEnv(alpha=-10, beta=0.5)
    price = 0.5
    profit = evaluate_profit(price, acceptance_model)
    true_prob = env.compute_acceptance_probability(price)
    true_profit = price * true_prob
    print("True Profit:", true_profit)
    print("Profit for price", price, ":", profit)
    # Assert that the profit is a numpy array of shape (1, 1)
    assert profit.shape == (1, 1), "Profit shape is incorrect."
    assert profit[0, 0].dtype == np.float32, "Profit should be a float."
    assert np.isclose(profit[0,0], true_profit, rtol=0.1), "Profit evaluation is incorrect."

    # Check with price array
    prices = np.linspace(0, 1, 11, endpoint=True)
    profits = evaluate_profit(prices, acceptance_model)
    assert profits.shape == (11, 1), "Profit shape is incorrect."
    assert profits.dtype == np.float32, "Profit should be a float."



def test_price_optimisation(data, acceptance_model):
    """Test the price optimisation function."""
    X, y, best_price, best_reward = data
    # Optimise prices
    initial_price = 0.1
    optimised_price = bayesian_optimise_price(
        acceptance_model,
        initial_price,
        learning_rate=0.01,
        num_iterations=100,
        verbose=True,
    )
    print("Optimised Price:", optimised_price)
    print("Best Price:", best_price)

    assert isinstance(optimised_price, float), "Optimised price should be a float."
    assert np.isclose(optimised_price, best_price, rtol=0.05), (
        "Price optimisation is incorrect."
    )
    assert optimised_price > 0, "Optimised price should be positive."


def test_robust_evaluation(acceptance_model):
    """Test the robust profit evaluation function."""
    price = 0.5
    delta = 1.
    robust_profit = evaluate_robust_profit(price, acceptance_model, delta)
    profit = evaluate_profit(price, acceptance_model)
    print("Expected Profit:", profit)
    print("Robust profit", price, ":", robust_profit)
    # Assert that the profit is a numpy array of shape (1, 1)
    assert profit.shape == (1, 1), "Profit shape is incorrect."
    assert profit[0, 0].dtype == np.float32, "Profit should be a float."
    assert robust_profit[0,0] <= profit, (
        "Robust profit should be less than or equal to the expected profit."
    )

    robust_profit_d0 = evaluate_robust_profit(price, acceptance_model, 0.0)
    print("Robust profit with delta=0:", robust_profit_d0)
    # Assert that the profit is a numpy array of shape (1, 1)
    assert profit.shape == (1, 1), "Profit shape is incorrect."
    assert np.isclose(robust_profit_d0[0,0], profit, rtol=0.05), (
        "Robust profit evaluation is incorrect."
    )

    # Check with price array
    prices = np.linspace(0, 1, 11, endpoint=True)
    profits = evaluate_robust_profit(prices, acceptance_model, delta)
    assert profits.shape == (11, 1), "Profit shape is incorrect."
    assert profits.dtype == np.float32, "Profit should be a float."


def test_robust_price_optimisation(data, acceptance_model):
    """Test the price optimisation function."""
    X, y, standard_price, standard_reward = data
    # Optimise prices
    initial_price = 0.1
    delta = 0.5
    robust_price = bayesian_optimise_robust_price(
        acceptance_model,
        initial_price,
        delta=delta,
        learning_rate=0.01,
        num_iterations=100,
        verbose=True,
    )
    print("Robust Price:", robust_price)
    print("Standard Price:", standard_price)

    assert isinstance(robust_price, float), "Optimised price should be a float."
    assert robust_price > 0, "Optimised price should be positive."

    # Assert that the robust profit at the robust price is less than or equal to the standard profit
    robust_profit_robust_price = evaluate_robust_profit(
        robust_price, acceptance_model, delta
    )
    standard_profit_robust_price = evaluate_profit(robust_price, acceptance_model)
    print("Robust profit at robust price:", robust_profit_robust_price)
    print("Standard profit at robust price:", standard_profit_robust_price)
    assert robust_profit_robust_price <= standard_profit_robust_price, (
        "Robust profit at robust price should be less than or equal to the standard profit."
    )

    robust_profit_standard_price = evaluate_robust_profit(
        standard_price, acceptance_model, delta
    )
    print("Robust profit at standard price:", robust_profit_standard_price)
    assert robust_profit_standard_price <= robust_profit_robust_price, (
        "Robust profit at standard price should be less than or equal to the standard profit."
    )


def test_evaluate_mle_profit(acceptance_model):
    """Test the profit evaluation function."""
    env = SimpleEnv(alpha=-10, beta=0.5)
    price = 0.5
    profit = evaluate_mle_profit(price, acceptance_model)
    true_prob = env.compute_acceptance_probability(price)
    true_profit = price * true_prob
    print("True Profit:", true_profit)
    print("Profit for price", price, ":", profit)
    # Assert that the profit is a numpy array of shape (1, 1)
    assert profit.shape == (1, 1), "Profit shape is incorrect."
    assert profit[0, 0].dtype == np.float32, "Profit should be a float."
    assert np.isclose(profit[0,0], true_profit, rtol=0.1), "Profit evaluation is incorrect."

    # Check with price array
    prices = np.linspace(0, 1, 11, endpoint=True)
    profits = evaluate_profit(prices, acceptance_model)
    assert profits.shape == (11, 1), "Profit shape is incorrect."
    assert profits.dtype == np.float32, "Profit should be a float."



def test_mle_price_optimisation(data, acceptance_model):
    """Test the price optimisation function."""
    X, y, best_price, best_reward = data
    # Optimise prices
    initial_price = 0.1
    optimised_price = bayesian_optimise_mle_price(
        acceptance_model,
        initial_price,
        learning_rate=0.01,
        num_iterations=100,
        verbose=True,
    )
    print("Optimised Price:", optimised_price)
    print("Best Price:", best_price)

    assert isinstance(optimised_price, float), "Optimised price should be a float."
    assert np.isclose(optimised_price, best_price, rtol=0.05), (
        "Price optimisation is incorrect."
    )
    assert optimised_price > 0, "Optimised price should be positive."