"""Test model based optimisation."""

import pytest
import numpy as np
import torch
from mbdro.optimisation_mb.standard_mb import evaluate_profit, optimise_price
from mbdro.optimisation_mb.dro_xi_mb import (
    evaluate_robust_profit,
    optimise_robust_price,
)
from mbdro.optimisation_mb.learn_conversion import train_model
from mbdro.envs.simple import SimpleEnv




@pytest.fixture(scope="module")
def data():
    """Fixture to generate data."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    env = SimpleEnv(alpha=-10, beta=0.5)
    X, y, r = env.generate_data(num_samples=500)

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
    acceptance_model, info = train_model(X, y)
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
    assert profit.shape == (1, 1), "Profit shape is incorrect."
    assert profit[0,0].dtype == np.float32, "Profit should be a float."
    assert np.isclose(profit[0,0], true_profit, rtol=0.05), "Profit evaluation is incorrect."

    # Check with price array
    prices = np.linspace(0, 1, 11, endpoint=True)
    profits = evaluate_profit(prices, acceptance_model)
    assert isinstance(profits, np.ndarray), "Profits should be a numpy array."
    assert profits.shape == (11, 1), "Profits shape is incorrect."


def test_price_optimisation(data, acceptance_model):
    """Test the price optimisation function."""
    X, y, best_price, best_reward = data
    # Optimise prices
    initial_price = 0.1
    optimised_price = optimise_price(
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
    env = SimpleEnv(alpha=-10, beta=0.5)
    price = 0.5
    delta = 0.2
    robust_profit = evaluate_robust_profit(price, acceptance_model, delta)
    true_prob = env.compute_acceptance_probability(price)
    true_profit = price * true_prob
    print("True Profit:", true_profit)
    print("Robust profit", price, ":", robust_profit)
    assert robust_profit.shape == (1, 1), "Robust profit shape is incorrect."
    assert robust_profit[0,0].dtype == np.float32, "Robust profit should be a float."
    assert robust_profit <= true_profit, (
        "Robust profit should be less than or equal to the true profit."
    )

    robust_profit_d0 = evaluate_robust_profit(price, acceptance_model, 0.0)
    print("Robust profit with delta=0:", robust_profit_d0)
    assert np.isclose(robust_profit_d0, true_profit, rtol=0.05), (
        "Robust profit evaluation is incorrect."
    )

    # Check with price array
    prices = np.linspace(0, 1, 11, endpoint=True)
    robust_profits = evaluate_robust_profit(prices, acceptance_model, delta)
    assert isinstance(robust_profits, np.ndarray), (
        "Robust profits should be a numpy array."
    )
    assert robust_profits.shape == (11, 1), "Robust profits shape is incorrect."


def test_robust_price_optimisation(data, acceptance_model):
    """Test the price optimisation function."""
    X, y, standard_price, standard_reward = data
    # Optimise prices
    initial_price = 0.1
    delta = 0.2
    robust_price = optimise_robust_price(
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



def test_profit_evaluate_array(acceptance_model):
    """Test the profit evaluation function."""
    prices = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
    profits = evaluate_profit(prices, acceptance_model)
    assert isinstance(profits, np.ndarray), "Profits should be a numpy array."
    assert profits.shape == (101,1), "Profits shape is incorrect."