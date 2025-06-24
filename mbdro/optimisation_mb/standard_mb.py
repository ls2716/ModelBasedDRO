"""Standard mean profit optimisation using logistic model
of the environment.

The optimisation is performed using PyTorch optimisation.
"""

import torch
import torch.optim as optim


def evaluate_profit(price, acceptance_model):
    """Evaluate the profit for a given price using the acceptance model."""
    price_pt = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
    # Compute the acceptance probability using the logistic model
    acceptance_prob = acceptance_model(price_pt)

    # Compute the profit as the product of price and acceptance probability
    profit = price_pt * acceptance_prob

    return profit.detach().numpy() # Convert to a numypy array for easier handling


def evaluate_profit_pt(price, acceptance_model):
    """Evaluate the profit for a given price using the acceptance model."""
    # Compute the acceptance probability using the logistic model
    acceptance_prob = acceptance_model(price)

    # Compute the profit as the product of price and acceptance probability
    profit = price * acceptance_prob

    return profit 


def optimise_price(
    acceptance_model,
    initial_price,
    learning_rate=0.01,
    num_iterations=100,
    verbose=False,
):
    """Optimise the price using gradient ascent."""
    # If the initial price is not a float raise an error
    if not isinstance(initial_price, float):
        raise ValueError("Initial price must be a float.")
    # Convert initial price to a PyTorch tensor and set requires_grad=True for optimisation
    price = torch.nn.Parameter(torch.tensor(initial_price, dtype=torch.float32))

    # Define the optimizer
    optimizer = optim.Adam([price], lr=learning_rate)

    for _ in range(num_iterations):
        # Zero the gradients
        optimizer.zero_grad()

        # Compute the profit
        profit = evaluate_profit_pt(price, acceptance_model)

        # Compute loss
        loss = - profit

        # Backpropagate the gradients
        loss.backward()

        # Update the price using the optimizer
        optimizer.step()

        if verbose:
            print(
                f"Iteration {_ + 1}/{num_iterations}, Loss: {loss.item()}, Price: {price.item()}"
            )

    return price.item()  # Return the optimised price as a Python float

