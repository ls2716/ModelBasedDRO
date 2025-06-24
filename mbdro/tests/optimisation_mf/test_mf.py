import pytest

import numpy as np
from mbdro.optimisation_mf.standard_mf import evaluate_profit_det, optimise_price_det
from mbdro.optimisation_mf.dro_xi_mf import evaluate_robust_profit_det, optimise_robust_price_det
from mbdro.envs.simple import SimpleEnv


@pytest.fixture(scope="module")
def data():
    np.random.seed(0)  # Set the random seed for reproducibility
    """Fixture to create synthetic data."""
    env = SimpleEnv(alpha=-10, beta=0.5)
    # Compute the optimal price
    prices = np.linspace(0, 1, 6, endpoint=True).reshape(-1, 1)
    acceptance_probabilities = env.compute_acceptance_probability(prices)
    outcomes = np.random.binomial(1, acceptance_probabilities)
    rewards = prices * acceptance_probabilities
    print("Acceptance Probabilities:\n", acceptance_probabilities.flatten())
    print("Rewards:\n", rewards.flatten())
    # Return the optimal price and the corresponding reward
    best_price_index = np.argmax(rewards)
    best_price = prices[best_price_index, 0]
    best_reward = rewards[best_price_index, 0]

    # Generate data for finite action space
    action_space = np.linspace(0, 1, 6, endpoint=True)
    prices, price_indices, outcomes, reward_outcomes = env.generate_data_finite_A(
        num_samples=2000, action_range=(0, 1), no_points=6
    )
    return (
        prices,
        price_indices,
        outcomes,
        action_space,
        best_price,
        best_reward,
        rewards,
    )


def test_price_optimisation(data):
    """Test the price optimisation function."""
    (
        prices,
        price_indices,
        outcomes,
        action_space,
        best_price,
        best_reward,
        all_profits,
    ) = data

    X = prices
    y = outcomes
    pi_0 = np.ones_like(action_space) / action_space.shape[0]
    # Compute the optimal price using model-free optimisation
    best_price_index, best_price, best_profit, profits = optimise_price_det(
        price_indices, X, y, action_space, pi_0
    )
    print("Estimated profits")
    print(profits)

    assert np.isclose(best_profit, best_reward, rtol=0.05), (
        f"Expected best profit: {best_reward}, but got: {best_profit}"
    )

    # Check if the best price is close to the expected best price
    assert np.isclose(action_space[best_price_index], best_price), (
        f"Expected best price: {best_price}, but got: {action_space[best_price_index]}"
    )


def test_profit_evaluation(data):
    """Test the profit evaluation function."""
    (
        prices,
        price_indices,
        outcomes,
        action_space,
        best_price,
        best_reward,
        all_profits,
    ) = data

    X = prices
    y = outcomes
    pi_0 = np.ones_like(action_space) / action_space.shape[0]

    # Compute the profit for a specific price index
    price_index = 2  # Example price index
    profit = evaluate_profit_det(price_index, price_indices, X, y, action_space, pi_0)

    assert np.isclose(profit, all_profits[price_index], rtol=0.05), (
        f"Expected profit: {all_profits[price_index]}, but got: {profit}"
    )


def test_robust_evaluation(data):
    """Test robust price evaluation."""
    (
        prices,
        price_indices,
        outcomes,
        action_space,
        best_price,
        best_reward,
        all_profits,
    ) = data
    X = prices
    y = outcomes
    delta = 0.5
    pi_0 = np.ones_like(action_space) / action_space.shape[0]
    # Compute the robust profit for a specific price index
    price_index = 2  # Example price index
    robust_profit = evaluate_robust_profit_det(
        price_index, price_indices, X, y, action_space, pi_0, delta
    )
    assert robust_profit <= all_profits[price_index], (
        f"Robust profit should be less than or equal to the original profit: {all_profits[price_index]}"
    )


def test_robust_optimisation(data):
    """Test robust optimisation"""
    (
        prices,
        price_indices,
        outcomes,
        action_space,
        best_price,
        best_reward,
        all_profits,
    ) = data
    X = prices
    y = outcomes
    delta = 0.5
    pi_0 = np.ones_like(action_space) / action_space.shape[0]
    # Compute the optimal price using model-free optimisation
    best_price_index, best_price, best_robust_profit, robust_profits = optimise_robust_price_det(
        price_indices, X, y, action_space, pi_0, delta=delta
    )
    all_profits = all_profits.flatten()
    print("Profits")
    print(all_profits)
    print("Estimated profits")
    print(robust_profits)
    assert (robust_profits<=all_profits).all(), "Robust profits should be less than or equal to the original profits"