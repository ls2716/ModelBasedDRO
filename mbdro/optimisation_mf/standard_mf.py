"""Standard mean profit optimisation for model-free optimisation.

In this module, we implement the standard mean profit optimisation for model-free optimisation.

This is a simple optimisation problem where we want to maximise the mean profit based
on discrete actions (prices) and reward signals (acceptance probabilities multiplied by prices).

The pricing function should find a single price that maximises the mean profit.
The optimisation is done by search as there is no gradient information.
"""

import numpy as np


def evaluate_profit_det(price_index, price_indices, X, y, action_space, pi_0):
    """Evaluate the profit for a given price using model-free evaluation."""
    # Compute the indicator variable for when price=X - use is close to avoid floating point issues
    indicator = price_index == price_indices
    # Compute the normalisation factor
    norm_factor = np.mean(indicator, axis=0)/pi_0[price_index]
    # Compute the reward estimate
    reward_estimate = np.mean(
        (action_space[price_index] * y * indicator) / pi_0[price_index], axis=0
    )
    # Compute the profit
    profit = reward_estimate / norm_factor
    return profit[0]


def optimise_price_det(price_indices, X, y, action_space, pi_0):
    """Optimise the price using model-free optimisation."""
    # Compute the profits for each price
    profits = np.zeros_like(action_space)
    for i in range(action_space.shape[0]):
        profits[i] = evaluate_profit_det(
            i, price_indices, X, y, action_space, pi_0
        )
    # Find the best price index
    best_price_index = np.argmax(profits)
    best_price = action_space[best_price_index]
    # Find the best profit
    best_profit = profits[best_price_index]
    return best_price_index, best_price, best_profit, profits