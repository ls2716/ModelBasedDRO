"""Implement standard optimisation in bayesian framework with respect to expected parameters MLE of alpha and beta."""

import torch
import torch.optim as optim


def evaluate_mle_profit(price, acceptance_model):
    """Evaluate robust profit for the given price using the acceptance model."""
    # Compute mle parameters
    alpha_mle = acceptance_model.alpha_mle
    beta_mle = acceptance_model.beta_mle
    price_pt = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
    robust_profit = evaluate_mle_profit_pt(
        price_pt, alpha_mle=alpha_mle, beta_mle=beta_mle
    )
    return robust_profit.detach().numpy() 


def evaluate_mle_profit_pt(price, alpha_mle, beta_mle):
    """Evaluate profit for the given price using the acceptance model."""
    # Compute the acceptance probability using the logistic model
    acceptance_prob = torch.sigmoid(alpha_mle * (price - beta_mle))

    # Compute profits
    profit = price * acceptance_prob

    return profit


def bayesian_optimise_mle_price(
    acceptance_model,
    initial_price,
    learning_rate=0.05,
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
    # Compute mle parameters
    alpha_mle = acceptance_model.alpha_mle
    beta_mle = acceptance_model.beta_mle

    # Run optimisation
    for i in range(num_iterations):
        optimizer.zero_grad()

        profit = evaluate_mle_profit_pt(price, alpha_mle=alpha_mle, beta_mle=beta_mle)
        # Compute the loss
        loss = -profit

        # Backpropagation
        loss.backward()
        optimizer.step()

        if verbose and i % 200 == 0:
            print(f"Iteration {i}, Profit: {profit.item()}")

    return price.item()
