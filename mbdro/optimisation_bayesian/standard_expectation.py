"""Implement standard optimisation in bayesian framework."""

import torch
import torch.optim as optim


def evaluate_profit(price, acceptance_model):
    """Evaluate profit for the given price using the acceptance model."""
    price_pt = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
    profit = evaluate_profit_pt(price_pt, acceptance_model)
    return profit.detach().numpy()


def evaluate_profit_pt(price, acceptance_model):
    """Evaluate profit for the given price using the acceptance model."""
    # Compute the acceptance probability using the logistic model
    acceptance_prob = acceptance_model(price)
    # Compute profits
    profits = price * acceptance_prob
    # Compute the expected profit
    profit = profits.mean(dim=1, keepdim=True)
    return profit


def bayesian_optimise_price(
    acceptance_model,
    initial_price,
    learning_rate=0.01,
    num_iterations=200,
    verbose=False,
):
    """Optimise robust price using xi^2 divergence."""
    if not isinstance(initial_price, (int, float)):
        raise ValueError("Initial price must be a scalar (int or float).")

    # Initialise the prices
    price = torch.nn.Parameter(
        torch.tensor(initial_price, dtype=torch.float32).reshape(-1, 1)
    )
    # Initialise the optimizer
    optimizer = optim.Adam([price], lr=learning_rate)

    # Run optimisation
    for i in range(num_iterations):
        optimizer.zero_grad()

        profit = evaluate_profit_pt(price, acceptance_model)
        # Compute the loss
        loss = -profit

        # Backpropagation
        loss.backward()
        optimizer.step()

        if verbose and i % 200 == 0:
            print(f"Iteration {i}, Profit: {profit.item()}")

    return price.item()
