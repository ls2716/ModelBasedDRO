import numpy as np


n_features = 2
n_actions = 5

def generate_data(parameter_seed, n_samples=1000, verbose=False, generation_seed=None):
    """
    Generate synthetic data for a voting experiment.

    Args:
        weights (list): Weights for the features.
        biases (list): Biases for the features.
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Generated data of shape (n_samples, n_features).
    """
    np.random.seed(parameter_seed)
    # Generate weigths and biases
    input_dim = n_features + n_actions
    weights = np.random.randn(input_dim)
    bias = np.random.randn()

    np.random.seed(generation_seed)

    # Step 1: Generate 2D features from a multivariate normal
    mean = np.zeros(n_features)
    cov = np.eye(n_features)  # Identity covariance for simplicity

    # Print weights and biases and mean if verbose
    if verbose:
        print(f"Weights: {weights}")
        print(f"Bias: {bias}")
        print(f"Mean: {mean}")

    X = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Step 2: Sample random actions from {0,1,2,3,4}
    actions = np.random.randint(0, n_actions, size=n_samples)

    # Step 3: One-hot encode actions
    A_one_hot = np.eye(n_actions)[actions]  # shape: (n_samples, 5)

    # Step 4: Define weights for logistic function
    # Concatenate features and one-hot actions, so total input dim = 2 + 5 = 7
    input_dim = n_features + n_actions

    # Step 5: Compute logits and apply sigmoid to get probabilities
    X_augmented = np.hstack([X, A_one_hot])  # shape: (n_samples, 7)
    logits = X_augmented @ weights + bias
    probs = 1 / (1 + np.exp(-logits))

    # Step 6: Sample labels from Bernoulli using the computed probabilities
    labels = np.random.binomial(1, probs)

    mean_probs = []
    # Print mean y for each action
    for action in range(n_actions):
        mean_y = np.mean(labels[actions == action])
        mean_probs.append(mean_y)
        if verbose:
            print(f"Mean y for action {action}: {mean_y}")

    return X, A_one_hot, labels, {
        "weights": weights,
        "bias": bias,
        "mean_probs": mean_probs,
    }


if __name__ == "__main__":
    
    seed = 0

    n_samples = 1000

    X, A_one_hot, labels, _ = generate_data(seed, n_samples, verbose=True)

