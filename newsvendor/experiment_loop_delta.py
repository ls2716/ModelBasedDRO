"""Run an experiment with check to see how the robust objetive changes with delta."""

import numpy as np
from generate_data import compute_reward, generate_data
from KL_DRO import optimise_KL_DRO, evaluate_KL_DRO_bs


def run_experiment(
    demand_samples, delta_arr, backorder_cost=0, holding_cost=0, order_cost=0
):
    """
    Run an experiment to evaluate the KL-DRO objective function for different delta values.

    Args:
        demand_samples (np.ndarray): Array of demand samples.
        delta_arr (list): List of delta values to evaluate.
        backorder_cost (float): Cost of backordering.
        holding_cost (float): Cost of holding inventory.
        order_cost (float): Cost of placing an order.

    Returns:
        list: List of KL-DRO objective values for each delta.
    """
    optimal_order_quantities = {}
    dro_rewards = {delta: {} for delta in delta_arr}


    # Compute order quantities and KL-DRO values for each delta
    for delta in delta_arr:
        # Compute the KL-DRO value
        _, _, optimal_dro_quantity, optimal_dro_reward = optimise_KL_DRO(
            demand_samples,
            delta,
            backorder_cost=backorder_cost,
            holding_cost=holding_cost,
            order_cost=order_cost,
        )
        optimal_order_quantities[delta] = float(optimal_dro_quantity)

    optimal_order_quantities_arr = np.array(
        [optimal_order_quantities[delta] for delta in delta_arr]
    )
    rewards = compute_reward(
        demand_samples,
        optimal_order_quantities_arr,
        backorder_cost=backorder_cost,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )
    for i, delta_ev in enumerate(delta_arr):
        # Compute the KL-DRO value for the standard order quantity
        for j, delta_opt in enumerate(delta_arr):
            # Compute the KL-DRO value
            reward, dro_reward, opt_dict = evaluate_KL_DRO_bs(rewards[j, :], delta=delta_ev)
            dro_rewards[delta_ev][delta_opt] = float(dro_reward)
            dro_rewards[0][delta_opt] = float(reward)
            # print(
            #     f"Delta: {delta_ev}, Order Quantity: {optimal_order_quantities[delta_opt]}, "
            #     f"Standard Reward: {reward}, KL-DRO Reward: {dro_reward}"
            # )
            # print(opt_dict)
            # input("Press Enter to continue...")

    return {
        "optimal_order_quantities": optimal_order_quantities,
        "dro_rewards": dro_rewards,
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 0
    np.random.seed(seed)
    # Set environment parameters
    backorder_cost = 8
    holding_cost = 4
    order_cost = 2
    # Generate demand samples
    num_samples = 10000
    mean_demand = 20
    std_demand = 50
    demand_samples = generate_data(num_samples, mean_demand, std_demand)

    # Define delta values to evaluate
    delta_arr = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    # Run the experiment
    results = run_experiment(
        demand_samples, delta_arr, backorder_cost, holding_cost, order_cost
    )

    # Print the results
    print("Optimal Order Quantities:", results["optimal_order_quantities"])
    print("Dro Rewards:", results["dro_rewards"])

    # Save the results to a json file
    import json
    import os

    # Create the directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    # Save the results to a JSON file
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results/experiment_results.json")
