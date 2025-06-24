"""Distributionally robust profit optimisation using logistic model
of the environment using Xi^2 divergence.

The optimisation is performed using PyTorch optimisation.
"""

import torch
import torch.optim as optim


def evaluate_robust_profit(price, acceptance_model, delta):
    """Evaluate the profit for a given price using the acceptance model."""
    price_pt = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
    # Compute the acceptance probability using the logistic model
    acceptance_prob = acceptance_model(price_pt)

    # Compute the theoretical mean
    mean_profit = price_pt * acceptance_prob
    # Compute the theoretical variance
    variance_profit = acceptance_prob * torch.pow(price_pt - mean_profit, 2) + (
        1 - acceptance_prob
    ) * torch.pow(mean_profit, 2)

    robust_profit = mean_profit - torch.sqrt(delta * variance_profit)

    return robust_profit.detach().numpy()  # Convert to numpy array


def evaluate_robust_profit_pt(price, acceptance_model, delta):
    """Evaluate the profit for a given price using the acceptance model."""
    # Compute the acceptance probability using the logistic model
    price = price.reshape(-1, 1)
    acceptance_prob = acceptance_model(price)

    # Compute the theoretical mean
    mean_profit = price * acceptance_prob
    # Compute the theoretical variance
    variance_profit = acceptance_prob * torch.pow(price - mean_profit, 2) + (
        1 - acceptance_prob
    ) * torch.pow(mean_profit, 2)

    robust_profit = mean_profit - torch.sqrt(delta * variance_profit)

    return robust_profit  # Convert to a Python float for easier handling


def optimise_robust_price(
    acceptance_model,
    initial_price,
    delta,
    learning_rate=0.01,
    num_iterations=100,
    verbose=False,
):
    """Optimise the price using gradient ascent."""
    if not isinstance(initial_price, (int, float)):
        raise ValueError("Initial price must be a number.")
    # Convert initial price to a PyTorch tensor and set requires_grad=True for optimisation
    price = torch.nn.Parameter(torch.tensor(initial_price, dtype=torch.float32))

    # Define the optimizer
    optimizer = optim.Adam([price], lr=learning_rate)

    for _ in range(num_iterations):
        # Zero the gradients
        optimizer.zero_grad()

        # Compute the profit
        profit = evaluate_robust_profit_pt(price, acceptance_model, delta)

        # Compute loss
        loss = -profit

        # Backpropagate the gradients
        loss.backward()

        # Update the price using the optimizer
        optimizer.step()

        if verbose and (_ + 1) % 10 == 0:
            print(
                f"Iteration {_ + 1}/{num_iterations}, Loss: {loss.item()}, Price: {price.item()}"
            )

    return price.item()  # Return the optimised price as a Python float
