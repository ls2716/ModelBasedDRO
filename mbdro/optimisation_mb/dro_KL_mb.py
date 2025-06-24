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
    robust_profit = evaluate_robust_profit_pt(price_pt, acceptance_model, delta)

    return robust_profit.detach().numpy()  # Convert to numpy array


def evaluate_robust_profit_pt(price, acceptance_model, delta):
    """Evaluate the profit for a given price using the acceptance model."""
    # Compute the acceptance probability using the logistic model
    price = price.reshape(-1, 1)

    #

    # Compute the robust profit
    alpha = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    # Initialise the optimiser for alpha
    optimizer = optim.Adam([alpha], lr=0.01)

    for i in range(100):
        optimizer.zero_grad()
        
        acceptance_prob = acceptance_model(price)
        # Compute the robust objective
        ro = (
            -alpha
            * torch.log(
                acceptance_prob * torch.exp(-price / alpha)
                + (1 - acceptance_prob) * torch.exp(0 / alpha)
                + 1e-8
            )
            - alpha * delta
        )
        loss = -ro.mean()
        # Backpropagation
        loss.backward()
        optimizer.step()

    robust_profit = (
        -alpha
        * torch.log(
            acceptance_prob * torch.exp(-price / alpha)
            + (1 - acceptance_prob) * torch.exp(0 / alpha)
            + 1e-8
        )
        - alpha * delta
    )

    return robust_profit


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
    alpha = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

    # Define the optimizer
    optimizer = optim.Adam([price, alpha], lr=learning_rate)

    for _ in range(num_iterations):
        # Zero the gradients
        optimizer.zero_grad()

        acceptance_prob = acceptance_model(price)

        # Compute the robust profit
        robust_profit = (
            -alpha
            * torch.log(
                acceptance_prob * torch.exp(-price / alpha)
                + (1 - acceptance_prob) * torch.exp(0 / alpha)
                + 1e-8
            )
            - alpha * delta
        )

        # Compute loss
        loss = -robust_profit

        # Backpropagate the gradients
        loss.backward()

        # Update the price using the optimizer
        optimizer.step()

        if verbose and (_ + 1) % 10 == 0:
            print(
                f"Iteration {_ + 1}/{num_iterations}, Loss: {loss.item()}, Price: {price.item()}, Alpha: {alpha.item()}"
            )
        if alpha.item() < 0.01:
            print("Alpha is too small, stopping optimisation.")
            break

    return price.item()  # Return the optimised price as a Python float
