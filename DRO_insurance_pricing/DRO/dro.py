"""Define the DRO with upsampling methods for the pricing strategy."""


import torch
from standard_optimisation import PricingModel
from torch import optim
from copy import deepcopy

def robust_objective(X, predicted_prices, acceptance_model, alpha, delta):
    # Get the original prices from the data
    prices = X[:, -1].view(-1, 1)
    # Create a tensor with features
    features = X[:, :-1]
    # Create a tensor with features and predicted prices
    X_in = torch.cat([features, predicted_prices], dim=1)
    # Compute the acceptance probability
    acceptance_prob = acceptance_model(X_in)
    expected_exprew_success = acceptance_prob * torch.exp(-(predicted_prices - prices)/alpha)
    expected_exprew_failure = (1 - acceptance_prob) * 1.0
    expected_exp_rew = expected_exprew_success + expected_exprew_failure
    expected_exp_rew = torch.mean(expected_exp_rew)
    # Compute the robust objectivecd 
    robust_objective = -alpha*torch.log(expected_exp_rew) - alpha*delta
    return robust_objective

def get_reward_distribution(X, predicted_prices, acceptance_model):
    """Compute the samples of the reward distribution."""
    samples = []
    # Get the original prices from the data
    prices = X[:, -1].view(-1, 1)
    # Get the features
    features = X[:, :-1]
    # Create a tensor with features and predicted prices
    X = torch.cat([features, predicted_prices], dim=1)
    # Compute the acceptance probability
    acceptance_prob = acceptance_model(X)
    # Compute the reward
    expected_exprew_success = predicted_prices - prices
    # To the samples, add int(100*acceptance_prob) samples of expected_exprew_success
    for i in range(X.shape[0]):
        samples += [expected_exprew_success[i].item()] * int(100*acceptance_prob[i].item())
    # To the samples, add int(100*(1-acceptance_prob)) samples of 1.0
    for i in range(X.shape[0]):
        samples += [0.0] * int(100*(1-acceptance_prob[i].item()))
    return samples


def robust_optimisation_loss(X, predicted_prices, acceptance_model, alpha, delta):
    # Compute the robust objective
    ro = robust_objective(X, predicted_prices, acceptance_model, alpha, delta)
    # return the loss
    return -ro



# Define the loss function for the pricing
def pricing_loss(X, pricing_model, acceptance_model, alpha, delta):
    # Get the features
    features = X[:, :-1]
    # Compute the prices
    predicted_prices = pricing_model(features)
    # Compute the robust objective
    ro = robust_objective(X, predicted_prices, acceptance_model, alpha, delta)
    # Compute the pricing loss
    loss = - ro
    return loss


def optimise_robust(data, acceptance_model, delta):
    """Run price optimisation using standard methods."""
    d = deepcopy(data)
    # Drop the Price, UnscaledPrice, Sale and cust_id columns
    d = data.drop(
        [ "UnscaledPrice", "Sale", "cust_id"], axis=1
    )

    # Get torch tensor from the train data
    X = torch.tensor(d.values, dtype=torch.float32)

    # Get the original prices from the data
    prices = X[:, -1].view(-1, 1)

    # Initialise the pricing model with the original prices as a vector
    pricing_model = PricingModel(prices)

    # Define the alpha parameter
    alpha = torch.tensor(0.4, requires_grad=True)

    # Define the optimizer for the pricing model
    optimizer = optim.Adam([pricing_model.prices, alpha], lr=0.002)

    # Train the pricing model
    for epoch in range(2000):
        # Forward pass
        loss = pricing_loss(X, pricing_model, acceptance_model, alpha, delta)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        # if loss is NaN, break the loop 
        if torch.isnan(loss):
            print("Loss is NaN, stopping training")
            break

    predicted_prices = pricing_model(X[:, :-1])

    # Save the results by creating new column in the train data
    d["Predicted_Price"] = predicted_prices.detach().numpy()
    return d


def evaluate_robust_profit(data, acceptance_model, delta, epochs=1000):
    # If cust_id, UnscaledPrice and Sale columns are present, drop them
    if "cust_id" in data.columns:
        data = data.drop(["cust_id", "UnscaledPrice", "Sale"], axis=1)

    predicted_prices = torch.tensor(
        data["Predicted_Price"].values, dtype=torch.float32
    ).view(-1, 1)
    # Drop the Predicted_Price, UnscaledPrice, Sale and cust_id columns
    d = data.drop("Predicted_Price", axis=1)

    # Get torch tensor from the data
    X = torch.tensor(d.values, dtype=torch.float32)

    # Define the alpha parameter
    alpha = torch.tensor(0.2, requires_grad=True)

    # Define the optimizer for the pricing model
    optimizer = optim.Adam([alpha], lr=0.002)

    # Train the pricing model
    for epoch in range(epochs):
        # Forward pass
        loss = -robust_objective(X, predicted_prices, acceptance_model, alpha, delta)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Alpha: {alpha.item()}")

    # Return the robust profit (loss)
    return -loss.item()
