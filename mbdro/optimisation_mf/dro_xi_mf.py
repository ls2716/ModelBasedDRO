"""Distributionally robust profit optimisation for model-free optimisation.

In this module, we implement the distributionally robust optimisation for model-free optimisation.

This is a simple optimisation problem where we want to maximise the mean profit based
on discrete actions (prices) and reward signals (acceptance probabilities multiplied by prices) under a distributionally robust framework in which the data distribution
can change within Xi^2 variation from the base distribution.

The pricing function should find a single price that maximises the robust profit.
The optimisation is done by search as there is no gradient information.
"""

import numpy as np


def evaluate_robust_profit_det(
    price_index, price_indices, X, y, action_space, pi_0, delta
):
    """Evaluate the profit for a given price using model-free evaluation."""
    # Compute the indicator variable for when price=X - use is close to avoid floating point issues
    indicator = price_index == price_indices
    # Compute the normalisation factor
    # norm_factor = np.mean(indicator, axis=0) / pi_0[price_index]
    # Collect the rewards where the price is equal to the action into a list
    rewards = []
    for i in range(len(X)):
        if indicator[i]:
            rewards.append(action_space[price_index] * y[i])
    # Transform into a numpy array
    rewards = np.array(rewards)
    # Compute mean profit using normalisation factor
    mean_profit = np.mean(rewards) 
    # Compute the variance of the rewards
    variance = np.var(rewards)

    # Compute the robust_profit
    robust_profit = mean_profit - np.sqrt(delta * variance)
    return robust_profit


def optimise_robust_price_det(price_indices, X, y, action_space, pi_0, delta):
    """Optimise the price using model-free optimisation."""
    # Compute the profits for each price
    profits = np.zeros_like(action_space)
    for i in range(action_space.shape[0]):
        profits[i] = evaluate_robust_profit_det(
            i, price_indices, X, y, action_space, pi_0, delta
        )
    # Find the best price index
    best_price_index = np.argmax(profits)
    best_price = action_space[best_price_index]
    # Find the best profit
    best_profit = profits[best_price_index]
    return best_price_index, best_price, best_profit, profits
