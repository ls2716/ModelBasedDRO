"""Implementation of a simple two-parameter pricing environment"""

import numpy as np


class SimpleEnv:
    """Simple pricing environment in which the acceptance probability is a logistic function."""

    def __init__(self, alpha=-10, beta=0.5):
        self.alpha = alpha  # Slope of the logistic function
        self.beta = beta  # Intercept for the logistic function

    def compute_acceptance_probability(self, price: float) -> float:
        """
        Compute the acceptance probability for a given price using the logistic function.

        Args:
            price (float): The price for which to compute the acceptance probability.

        Returns:
            float: Acceptance probability.
        """
        # Logistic function
        return 1 / (1 + np.exp(-self.alpha * (price - self.beta)))

    def generate_data(
        self, num_samples: int, price_range: tuple = (0, 1)
    ) -> np.ndarray:
        """
        Generate synthetic data for the environment.

        Args:
            num_samples (int): Number of samples to generate.
            price_range (tuple): Range of prices to generate.
                Defaults to (0, 1). (Sampled uniformly from this range)

        Returns:
            np.ndarray: Generated data samples.
        """
        # Generate prices from uniform distribution within the specified range
        prices = np.random.uniform(price_range[0], price_range[1], num_samples)
        # Generate acceptance probabilities using the logistic function
        acceptance_probabilities = self.compute_acceptance_probability(prices)
        # Generate binary outcomes based on acceptance probabilities
        outcomes = np.random.binomial(1, acceptance_probabilities)
        # Compute rewards based on the outcomes and prices
        rewards = prices * outcomes

        return prices.reshape(-1, 1), outcomes.reshape(-1, 1), rewards.reshape(
            -1, 1
        )

    def generate_data_finite_A(
        self, num_samples: int, action_range=(0, 1), no_points=21
    ):
        """Generate data points for a finite action space.

        Args:
            num_samples (int): Number of samples to generate.
            action_range (tuple): Range of actions to generate.
            no_points (int): Number of points to generate.
        Returns:
            tuple: Generated prices, price indices and outcomes.
                np.ndarray of shape (num_samples, 1): Generated prices.
                np.ndarray of shape (num_samples, 1): Indices of the generated prices.
                np.ndarray of shape (num_samples, 1): Generated outcomes.
        """
        # Generate prices by sampling uniformly from the action space
        action_space = np.linspace(
            action_range[0], action_range[1], no_points, endpoint=True
        )
        price_indices = np.random.choice(len(action_space), num_samples)
        prices = action_space[price_indices]
        # Generate acceptance probabilities using the logistic function
        acceptance_probabilities = self.compute_acceptance_probability(prices)
        # Generate binary outcomes based on acceptance probabilities
        outcomes = np.random.binomial(1, acceptance_probabilities)
        # Compute rewards based on the outcomes and prices
        rewards = prices * outcomes
        return (
            prices.reshape(-1, 1),
            price_indices.reshape(-1, 1),
            outcomes.reshape(-1, 1),
            rewards.reshape(-1, 1),
        )


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(0)
    # Example usage
    env = SimpleEnv()
    prices, outcomes = env.generate_data(num_samples=10)
    print("Generated Prices:", prices[:5])
    print("Generated Outcomes:", outcomes[:5])

    import matplotlib.pyplot as plt

    # Plot the acceptance probability as a function of price
    price_range = np.linspace(0, 1, 100)
    acceptance_probabilities = [
        env.compute_acceptance_probability(price) for price in price_range
    ]
    plt.plot(price_range, acceptance_probabilities)
    plt.plot(price_range, acceptance_probabilities * price_range)
    plt.xlabel("Price")
    plt.ylabel("Acceptance Probability")
    plt.title("Acceptance Probability vs Price")
    plt.grid()
    plt.show()
