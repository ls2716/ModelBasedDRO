"""Compute the optimal order quantity using KL-DRO."""

import numpy as np
from generate_data import compute_reward, generate_data

import matplotlib.pyplot as plt


def objective_KL_DRO(rewards, delta, alpha):
    """
    Compute the KL-DRO objective function.

    Args:
        rewards (np.ndarray): Array of rewards.
        delta (float): KL divergence threshold.
        alpha (float): Parameter for the KL-DRO objective.

    Returns:
        float: KL-DRO objective value.
    """
    exp_rewards = np.exp(-rewards / alpha)
    return -alpha * np.log(np.mean(exp_rewards)) - alpha * delta


def optimise_concave(func, left, right, verbose=False):
    """
    Optimise a concave function using binary search.

    Args:
        func (callable): Function to optimise.
        left (float): Left boundary of the search interval.
        right (float): Right boundary of the search interval.

    Returns:
        float: Optimal value of the function.
    """
    mid1 = left + (right - left) / 3
    mid2 = right - (right - left) / 3
    f1 = func(mid1)
    f2 = func(mid2)
    while abs(right - left) > 1e-5:
        if f1 < f2:
            left = mid1
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            f1 = func(mid1)
            f2 = func(mid2)
        else:
            right = mid2
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            f1 = func(mid1)
            f2 = func(mid2)
    # Print the left and right boundaries

    return f1, {"left": left, "right": right, "mid1": mid1, "mid2": mid2}


def evaluate_KL_DRO_bs(rewards, delta):
    """Evaluate the KL-DRO objective function using binary search.
    Args:
        rewards (np.ndarray): Array of rewards.
        delta (float): KL divergence threshold.
    Returns:
        float: KL-DRO objective value.
    """
    # Check if delta is 0
    if delta == 0:
        return np.mean(rewards,), np.mean(rewards), {}
    factor = 100
    left_alpha = 0.1
    right_alpha = 50
    # Compute the KL-DRO value
    dro_reward, optimisation_dict = optimise_concave(
            lambda alpha: objective_KL_DRO(rewards / factor, delta, alpha),
            left_alpha,
            right_alpha,
    )
    dro_reward = dro_reward * factor


    return np.mean(rewards), dro_reward, optimisation_dict


def evaluate_KL_DRO_samples(order_quantity, demand_samples, delta, backorder_cost=0, holding_cost=0, order_cost=0):
    # order_quantity = np.array(order_quantity)
    rewards = compute_reward(
        demand_samples,
        order_quantity,
        backorder_cost=backorder_cost,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )
    # Compute the KL-DRO value
    mean_reward, dro_reward, optimisation_dict = evaluate_KL_DRO_bs(
        rewards, delta
    )
    var_reward = np.std(rewards)
    return mean_reward, var_reward, dro_reward, optimisation_dict


def optimise_KL_DRO(
    demand_samples, delta, backorder_cost=0, holding_cost=0, order_cost=0
):
    """
    Compute the optimal order quantity using KL-DRO.

    Args:
        demand_samples (np.ndarray): Array of demand samples.
        delta (float): KL divergence threshold.

    Returns:
        float: Optimal order quantity.
    """
    order_quantities = np.arange(0, 100, 1)

    # Generate the rewards
    rewards = compute_reward(
        demand_samples,
        order_quantities,
        backorder_cost=backorder_cost,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )
    # Compute the optimal order quantity based on the rewards
    mean_reward = np.mean(rewards, axis=1)
    optimal_order_quantity = order_quantities[np.argmax(mean_reward)]
    optimal_standard_reward = mean_reward.max()

    kl_dro_values = np.array(
        [
            evaluate_KL_DRO_bs(rewards[i, :], delta=delta)[1]
            for i in range(len(order_quantities))
        ]
    )

    # Find the order quantity that maximizes the KL-DRO value
    optimal_kl_dro_order_quantity = order_quantities[np.argmax(kl_dro_values)]
    optimal_kl_dro_value = kl_dro_values.max()

    # # Plot the reward distribution for the optimal order quantity and the optimal KL-DRO order quantity
    # print(f"Optimal order quantity: {optimal_order_quantity}")
    # print(f"Optimal KL-DRO order quantity: {optimal_kl_dro_order_quantity}")

    # print(f"Optimal standard reward: {optimal_standard_reward}")
    # print(f"Optimal KL-DRO value: {optimal_kl_dro_value}")

    # plt.figure(figsize=(10, 6))
    # plt.hist(
    #     rewards[np.argmax(mean_reward)],
    #     bins=30,
    #     density=True,
    #     alpha=0.2,
    #     color="g",
    #     label="Standard Reward Distribution",
    # )
    # plt.hist(
    #     rewards[np.argmax(kl_dro_values)],
    #     bins=30,
    #     density=True,
    #     alpha=0.2,
    #     color="b",
    #     label="KL-DRO Reward Distribution",
    # )

    # plt.legend()
    # plt.xlabel("Reward")
    # plt.ylabel("Density")
    # plt.title("Reward Distribution for Optimal Order Quantities")
    # plt.grid()
    # plt.show()

    return (
        optimal_order_quantity,
        optimal_standard_reward,
        optimal_kl_dro_order_quantity,
        optimal_kl_dro_value,
    )


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    # Parameters for the data generation
    num_samples = 10000
    mean_demand = 20
    std_demand = 10

    # Generate the demand data
    demand_samples = generate_data(num_samples, mean_demand, std_demand)

    # Set the KL divergence threshold
    delta = 0.1

    # Compute the optimal order quantity using KL-DRO
    optimise_KL_DRO(
        demand_samples, delta, backorder_cost=8, holding_cost=4, order_cost=2
    )
