"""Define the standard optimisation methods for the pricing strategy."""

import torch
import torch.optim as optim
from copy import deepcopy


class PricingModel:
    def __init__(self, prices):
        """Initialise the pricing model."""
        self.prices = prices.clone().detach().view(-1, 1).requires_grad_(True)

    def forward(self, X):
        return self.prices

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


def evaluate_mean_profit(data, acceptance_model):
    
    """Evaluate the mean profit of the pricing strategy."""
    # Remove the cust_id, Sale and UnscaledPrice columns if they exist
    if "UnscaledPrice" in data.columns:
        data = data.drop(["UnscaledPrice", "Sale", "cust_id"], axis=1)

    predicted_prices = torch.tensor(
        data["Predicted_Price"].values, dtype=torch.float32
    ).view(-1, 1)
    # Drop the Predicted_Price, UnscaledPrice, Sale and cust_id columns
    d = data.drop("Predicted_Price", axis=1)

    # Get torch tensor from the data
    X = torch.tensor(d.values, dtype=torch.float32)
    # Get the original prices from the data
    prices = X[:, -1].view(-1, 1)
    # Get the features
    features = X[:, :-1]
    # Create a tensor with features and predicted prices
    X = torch.cat([features, predicted_prices], dim=1)
    # Compute the acceptance probability
    acceptance_prob = acceptance_model(X)
    # Compute the expected profit
    expected_profit = acceptance_prob * (predicted_prices - prices)
    # Compute the mean profit
    mean_profit = expected_profit.mean()
    return mean_profit.item(), acceptance_prob.mean().item()


# Define the loss function for the pricing
def pricing_loss(X, pricing_model, acceptance_model, min_acceptance_prob=0.0):
    # Get the original prices from the data
    prices = X[:, -1].view(-1, 1)
    # Get the features
    features = X[:, :-1]
    # Compute the prices
    predicted_prices = pricing_model(features)
    # Create a tensor with features and predicted prices
    X = torch.cat([features, predicted_prices], dim=1)
    # Compute the acceptance probability
    acceptance_prob = acceptance_model(X)
    mean_acceptance_prob = acceptance_prob.mean()
    # Compute the expected profit
    expected_profit = acceptance_prob * (predicted_prices - prices)
    # Compute the pricing loss
    loss = -expected_profit.mean() + torch.relu(
        min_acceptance_prob - mean_acceptance_prob
    )
    return loss


def optimise_standard(data, acceptance_model, min_acceptance_prob=0.0):
    """Run price optimisation using standard methods."""
    # Drop the Price, UnscaledPrice, Sale and cust_id columns
    d = deepcopy(data)
    d = d.drop(["UnscaledPrice", "Sale", "cust_id"], axis=1)

    # Get torch tensor from the train data
    X = torch.tensor(d.values, dtype=torch.float32)

    # Get the original prices from the data
    prices = X[:, -1].view(-1, 1)

    # Initialise the pricing model with the original prices as a vector
    pricing_model = PricingModel(prices)

    # Define the optimizer for the pricing model
    optimizer = optim.Adam([pricing_model.prices], lr=0.002)

    # Train the pricing model
    for epoch in range(1000):
        # Forward pass
        loss = pricing_loss(X, pricing_model, acceptance_model, min_acceptance_prob)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    predicted_prices = pricing_model(X[:, :-1])

    # Save the results by creating new column in the train data
    d["Predicted_Price"] = predicted_prices.view(-1, 1).detach().numpy()
    return d
