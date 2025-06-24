"""Conversion model for the acceptance model to be used in the MBRO algorithm."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LogisticModel(nn.Module):
    """Logistic regression model for acceptance probability.
    
    Callable with a price tensor to compute the acceptance probability.
    """

    def __init__(self):
        super(LogisticModel, self).__init__()
        # Initialize the parameters alpha and beta
        self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        return torch.sigmoid(self.alpha * (x - self.beta))


def train_model(
    X, y, lr=0.1, no_iterations=1000, patience=20, tolerance=0.0001, verbose=False, synthetic=True
):
    """Train the logistic regression model.
    
    Arguments:
        X -- Input data (prices) shape=(-1,1).
        y -- Target data (labels) shape=(-1,1).
        lr -- Learning rate for the optimizer.
        no_iterations -- Number of iterations for training.
        patience -- Number of iterations to wait before early stopping.
        tolerance -- Tolerance for early stopping.
        verbose -- Whether to print the loss during training.
    
    Returns:
        model -- Trained logistic regression model. (callable with a price tensor).
        info -- Dictionary with training information (loss, early_stop, iterations).
    """
    if synthetic:
        # Add 20 observations at price -10 and acceptance probability 1
        X = np.concatenate((X, np.full((20, 1), -5)), axis=0)
        y = np.concatenate((y, np.full((20, 1), 1)), axis=0)
        # Add 20 observations at price 10 and acceptance probability 0
        X = np.concatenate((X, np.full((20, 1), 5)), axis=0)
        y = np.concatenate((y, np.full((20, 1), 0)), axis=0)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


    # Initialize the model, loss function and optimizer
    model = LogisticModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_loss = np.inf
    stop_count = 0
    early_stop = False

    # Training loop
    for it in range(no_iterations):
        # Split the data into batches
        batch_size = 1000
        num_batches = int(np.ceil(len(X_tensor) / batch_size))
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(X_tensor))
            X_batch = X_tensor[start:end]
            y_batch = y_tensor[start:end]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # # Forward pass
        # outputs = model(X_tensor)
        # loss = criterion(outputs, y_tensor)

        # # Backward pass and optimization
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()]

        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        # Check for early stopping
        optimizer.zero_grad()

        if loss.item() < min_loss - tolerance:
            stop_count = 0
        else:
            stop_count += 1
            if stop_count >= patience:
                early_stop = True
                break
        
        if loss.item() < min_loss - tolerance:
            min_loss = loss.item()
        
        if verbose and it % 100 == 0:
            print(f"Iteration {it}, Loss: {loss.item()}")

    return model, {"loss": loss.item(), "early_stop": early_stop, "iterations": it}


if __name__=="__main__":

    from mbdro.envs.simple import SimpleEnv
    import numpy as np
    import matplotlib.pyplot as plt


    # Set seed for reproducibility
    np.random.seed(0)
    # Set seed for reproducibility
    torch.manual_seed(0)

    env = SimpleEnv(alpha=-10, beta=0.5)
    X, y = env.generate_data(num_samples=100)
    # Train the model
    model, info = train_model(X, y, lr=0.1, no_iterations=1000, verbose=True, patience=100, tolerance=0.000001)

    # Print the training info
    print("Training info:", info)
    # Print the model parameters
    print("Model parameters:")
    for param in model.parameters():
        print(param)


    # Plot the acceptance probability given the fitted model
    price_range = np.linspace(0,1, 100).reshape(-1, 1)
    price_tensor = torch.tensor(price_range, dtype=torch.float32)
    acceptance_prob = model(price_tensor).detach().numpy()

    # Compute the probability given the true model
    acceptance_prob_true = env.compute_acceptance_probability(price_range)

    plt.title("Acceptance Probability vs Price")
    plt.xlabel("Price")
    plt.ylabel("Acceptance Probability")
    plt.plot(price_range, acceptance_prob, label="Fitted Model")
    plt.plot(price_range, acceptance_prob_true, label="True Model")
    plt.legend()
    plt.grid()
    plt.show()
