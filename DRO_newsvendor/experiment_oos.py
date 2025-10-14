"""Run experiments for out-of-sample evaluation of the robust optimisation
method.
"""

import os
import json
import sys
import time
import numpy as np
import pandas as pd
import tqdm

from generate_data import generate_data
from train_model import train_model
from KL_DRO import optimise_KL_DRO, evaluate_KL_DRO_bs, evaluate_KL_DRO_samples


def run_single(no_samples, env_params, delta, evaluation_samples, verbose=False):
    """Run single experiment for out-of-sample evaluation of the robust
    optimisation method.
    """
    backorder_cost = env_params["backorder_cost"]
    holding_cost = env_params["holding_cost"]
    order_cost = env_params["order_cost"]
    mean_demand = env_params["mean_demand"]
    std_demand = env_params["std_demand"]
    # Generate demand samples
    demand_samples = generate_data(no_samples, mean_demand, std_demand)

    # Train the model
    mu_mle, sigma_mle = train_model(demand_samples)
    no_model_samples = 10000
    # print(f"Trained model parameters: mu = {mu_mle}, sigma = {sigma_mle}")

    # Generate model samples
    model_samples = generate_data(no_model_samples, mu_mle, sigma_mle)

    # Compute the optimal order quantity using data-based KL-DRO
    (
        data_optimal_order_quantity,
        data_optimal_standard_reward,
        data_optimal_kl_dro_order_quantity,
        data_optimal_kl_dro_value,
    ) = optimise_KL_DRO(
        demand_samples,
        delta,
        backorder_cost=backorder_cost,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )
    # Compute the rewards for the model samples
    (
        model_optimal_order_quantity,
        model_optimal_standard_reward,
        model_optimal_kl_dro_order_quantity,
        model_optimal_kl_dro_value,
    ) = optimise_KL_DRO(
        model_samples,
        delta,
        backorder_cost=backorder_cost,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )

    # Evaluate the mean reward in the true distribution
    model_mean_reward, model_var_reward, model_dro_reward, _ = evaluate_KL_DRO_samples(
        model_optimal_kl_dro_order_quantity,
        evaluation_samples,
        delta=0,
        backorder_cost=backorder_cost,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )
    data_mean_reward, data_var_reward, data_dro_reward, _ = evaluate_KL_DRO_samples(
        data_optimal_kl_dro_order_quantity,
        evaluation_samples,
        delta=0,
        backorder_cost=backorder_cost,
        holding_cost=holding_cost,
        order_cost=order_cost,
    )

    naive_data_mean_reward, naive_data_var_reward, naive_data_dro_reward, _ = (
        evaluate_KL_DRO_samples(
            data_optimal_order_quantity,
            evaluation_samples,
            delta=0,
            backorder_cost=backorder_cost,
            holding_cost=holding_cost,
            order_cost=order_cost,
        )
    )

    naive_model_mean_reward, naive_model_var_reward, naive_model_dro_reward, _ = (
        evaluate_KL_DRO_samples(
            model_optimal_order_quantity,
            evaluation_samples,
            delta=0,
            backorder_cost=backorder_cost,
            holding_cost=holding_cost,
            order_cost=order_cost,
        )
    )

    # Print results
    verbose = False
    if verbose:
        print(
            f"Data-based KL-DRO: Order Quantity: {data_optimal_kl_dro_order_quantity}, "
            f"Standard Reward: {data_mean_reward}, KL-DRO Reward: {data_dro_reward}"
        )
        print(
            f"Model-based KL-DRO: Order Quantity: {model_optimal_kl_dro_order_quantity}, "
            f"Standard Reward: {model_mean_reward}, KL-DRO Reward: {model_dro_reward}"
        )
        print(
            f"Naive data: Order Quantity: {data_optimal_order_quantity}, "
            f"Standard Reward: {naive_data_mean_reward}, KL-DRO Reward: {naive_data_dro_reward}"
        )
        print(
            f"Naive model: Order Quantity: {model_optimal_order_quantity}, "
            f"Standard Reward: {naive_model_mean_reward}, KL-DRO Reward: {naive_model_dro_reward}"
        )
        # Print the standard deviation of the rewards
        print("Data-based KL-DRO: Std of Reward: ", data_var_reward)
        print("Model-based KL-DRO: Std of Reward: ", model_var_reward)
        print("Naive data: Std of Reward: ", naive_data_var_reward)
        print("Naive model: Std of Reward: ", naive_model_var_reward)
        input(
            "Press Enter to continue to the next experiment..."
        )
    return (
        data_mean_reward,
        data_var_reward,
        model_mean_reward,
        model_var_reward,
        naive_data_mean_reward,
        naive_data_var_reward,
        naive_model_mean_reward,
        naive_model_var_reward,
    )


def loop(no_sims, no_samples, env_params, delta, evaluation_samples):
    """Run multiple experiments for out-of-sample evaluation of the robust
    optimisation method.
    """

    data_rewards = []
    data_var_rewards = []
    model_rewards = []
    model_var_rewards = []
    naive_data_rewards = []
    naive_data_var_rewards = []
    naive_model_rewards = []
    naive_model_var_rewards = []
    for i in tqdm.tqdm(range(no_sims)):
        # Run the experiment
        (
            data_mean_reward,
            data_var_reward,
            model_mean_reward,
            model_var_reward,
            naive_data_mean_reward,
            naive_data_var_reward,
            naive_model_mean_reward,
            naive_model_var_reward,
        ) = run_single(
            no_samples,
            env_params,
            delta,
            evaluation_samples,
        )

        data_rewards.append(data_mean_reward)
        data_var_rewards.append(data_var_reward)
        model_rewards.append(model_mean_reward)
        model_var_rewards.append(model_var_reward)
        naive_data_rewards.append(naive_data_mean_reward)
        naive_data_var_rewards.append(naive_data_var_reward)
        naive_model_rewards.append(naive_model_mean_reward)
        naive_model_var_rewards.append(naive_model_var_reward)

    return (
        data_rewards,
        data_var_rewards,
        model_rewards,
        model_var_rewards,
        naive_data_rewards,
        naive_data_var_rewards,
        naive_model_rewards,
        naive_model_var_rewards,
    )


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 0
    np.random.seed(seed)
    # Set environment parameters
    env_params = {
        "backorder_cost": 8,
        "holding_cost": 4,
        "order_cost": 2,
        "mean_demand": 10,
        "std_demand": 10,
    }

    # Set parameters for the experiment
    no_samples = 20
    no_evaluation_samples = 10000

    # Sample evaluation samples
    evaluation_samples = generate_data(
        no_evaluation_samples, env_params["mean_demand"], env_params["std_demand"]
    )

    # Set delta
    delta = 0.1
    no_sims = 200

    (
        data_rewards,
        data_var_rewards,
        model_rewards,
        model_var_rewards,
        naive_data_rewards,
        naive_data_var_rewards,
        naive_model_rewards,
        naive_model_var_rewards,
    ) = loop(
        no_sims=no_sims,
        no_samples=no_samples,
        env_params=env_params,
        delta=delta,
        evaluation_samples=evaluation_samples,
    )

    # Save the results to a json file
    results = {
        "data_rewards": data_rewards,
        "data_var_rewards": data_var_rewards,
        "model_rewards": model_rewards,
        "model_var_rewards": model_var_rewards,
        "naive_data_rewards": naive_data_rewards,
        "naive_data_var_rewards": naive_data_var_rewards,
        "naive_model_rewards": naive_model_rewards,
        "naive_model_var_rewards": naive_model_var_rewards,
        "delta": delta,
        "no_samples": no_samples,
    }

    # Save to the results folder with delta and no_samples in the name
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(
        results_folder,
        f"results_delta_{delta}_no_samples_{no_samples}.json",
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    # Print the mean and std of the rewards
    print(
        f"Data-based KL-DRO: Mean Reward: {np.mean(data_rewards)}, "
        f"Std Reward: {np.std(data_rewards)}"
    )
    print(
        f"Model-based KL-DRO: Mean Reward: {np.mean(model_rewards)}, "
        f"Std Reward: {np.std(model_rewards)}"
    )
    print(
        f"Naive data: Mean Reward: {np.mean(naive_data_rewards)}, "
        f"Std Reward: {np.std(naive_data_rewards)}"
    )
    print(
        f"Naive model: Mean Reward: {np.mean(naive_model_rewards)}, "
        f"Std Reward: {np.std(naive_model_rewards)}"
    )
    print('-------------------------------')
    print("Standard deviatins of the rewards:")
    print(
        f"Data-based KL-DRO: Mean Std: {np.mean(data_var_rewards)}"
    )
    print(
        f"Model-based KL-DRO: Mean Std: {np.mean(model_var_rewards)}"
    )
    print(
        f"Naive data: Mean Std: {np.mean(naive_data_var_rewards)}"
    )
    print(
        f"Naive model: Mean Std: {np.mean(naive_model_var_rewards)}"
    )

    
