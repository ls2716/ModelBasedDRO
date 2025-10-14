import numpy as np


def generate_data(num_samples, mean_demand, std_demand):
    """
    Generate demand data from a normal distribution.

    Args:
        num_samples (int): Number of samples to generate.
        mean_demand (float): Mean of the demand distribution.
        std_demand (float): Standard deviation of the demand distribution.

    Returns:
        np.ndarray: Array of generated demand samples.
    """

    demand_samples = np.random.normal(mean_demand, std_demand, num_samples * 3)
    # Remove all negative values
    demand_samples = demand_samples[demand_samples > 0]

    # If there are not enough samples, create more samples
    while len(demand_samples) < num_samples:
        demand_samples = np.concatenate(
            (
                demand_samples,
                np.random.normal(mean_demand, std_demand, num_samples * 3),
            )
        )
        # Remove all negative values
        demand_samples = demand_samples[demand_samples > 0]

    # If there are too many samples, truncate the array
    if len(demand_samples) > num_samples:
        demand_samples = demand_samples[:num_samples]

    return demand_samples


def compute_reward(
    demand_samples, order_quantity, backorder_cost=0, holding_cost=0, order_cost=0
):
    """
    Compute the reward based on demand samples and order quantity.

    Args:
        demand_samples (np.ndarray): Array of demand samples.
        order_quantity (np.ndarray): Order quantity.
        backorder_cost (float): Cost of backordering.
        holding_cost (float): Cost of holding inventory.
        order_cost (float): Cost of placing an order.

    Returns:
        float: Computed reward.
    """
    # Reshape the demand samples so that the result is a 2D array
    demand_samples = demand_samples.reshape(1, -1)
    order_quantity = order_quantity.reshape(-1, 1)

    # Calculate the negative cost
    cost = (
        np.maximum(demand_samples - order_quantity, 0) * backorder_cost
        + np.maximum(order_quantity - demand_samples, 0) * holding_cost
        + order_cost
    )
    return -cost


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    # Parameters for the data generation
    num_samples = 1000
    mean_demand = 20
    std_demand = 50

    # Generate the demand data
    demand_samples = generate_data(num_samples, mean_demand, std_demand)

    # Plot the generated demand data
    import matplotlib.pyplot as plt

    plt.hist(demand_samples, bins=30, density=True, alpha=0.6, color="g")
    plt.title("Generated Demand Samples")
    plt.xlabel("Demand")
    plt.ylabel("Density")
    plt.grid()
    plt.show()


    # Example usage of compute_reward
    order_quantity = np.array([10, 20, 30])
    rewards = compute_reward(
        demand_samples, order_quantity, backorder_cost=1, holding_cost=2, order_cost=3
    )

    # Plot the distibution of rewards for each order quantity
    for i, order in enumerate(order_quantity):
        plt.hist(
            rewards[i,:],
            bins=30,
            density=True,
            alpha=0.6,
            label=f"Order Quantity: {order}",
        )
    plt.title("Distribution of Rewards for Different Order Quantities")
    plt.xlabel("Reward")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()
    
