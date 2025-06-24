from generate_data import generate_data
from true_evaluation import optimise_policy
from policy import NeuralNetworkPolicy
from evaluation_analysis_DRO import (
    apply_policy,
    evaluate_policy_model_dro,
    evaluate_policy_data_dro,
)
from true_evaluation import (
    true_evaluation_data,
    true_evaluation,
    true_evaluation_data_dro,
    true_evaluation_dro,
)

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


def main(seed, generation_seed, delta=0.1):
    """Main function to run the full analysis.

    Args:
        seed (int): Random seed for reproducibility.
    """
    no_samples_for_policy = 10000
    input_dim = 2
    n_actions = 5
    no_sims = 500
    n_samples_arr = [
        300,
        500,
        1000,
        2000,
        3000,
        5000,
        10000,
        20000,
        50000,
    ]

    # Create a results directory for the seed
    results_dir = f"results_dro/seed_{seed}_{generation_seed}_{delta}"
    os.makedirs(results_dir, exist_ok=True)

    # Set the random seed for reproducibility
    torch.manual_seed(generation_seed)

    # Generate the parameters
    _, _, _, gen_dict = generate_data(
        seed, 1000000, verbose=True, generation_seed=generation_seed
    )

    # Get the weights and biases
    weights = gen_dict["weights"]
    bias = gen_dict["bias"]
    means = np.array(gen_dict["mean_probs"]).tolist()

    # Compute action costs as means round to 2 decimal places
    action_costs = [round(mean, 2) for mean in means]
    print(f"Action costs: {action_costs}")

    # Print the weights and biases to a file in the results directory
    with open(os.path.join(results_dir, "weights_bias.txt"), "w") as f:
        f.write(f"Weights: {weights}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Means: {means}\n")
        f.write(f"Action Costs: {action_costs}\n")

    # Try to load the policy from a file
    policy_file = os.path.join(results_dir, "policy.pth")
    policy = NeuralNetworkPolicy(
        input_dim=input_dim,
        output_dim=n_actions,
    )
    if os.path.exists(policy_file):
        policy.load_state_dict(torch.load(policy_file))
        print(f"Policy loaded from {policy_file}")
    else:
        print("No existing policy found. Training a new one.")
        mean_reward, policy = optimise_policy(
            seed, action_costs, no_samples_for_policy, generation_seed=generation_seed
        )

        print(f"Mean reward: {mean_reward}")

        # Save the policy to a file
        policy_file = os.path.join(results_dir, "policy.pth")
        torch.save(policy.state_dict(), policy_file)
        print(f"Policy saved to {policy_file}")

    # Evaluate the policy using true evaluation
    mean_reward_true = true_evaluation(seed, policy, action_costs, no_samples=1000000)
    print(f"Mean reward from true evaluation: {mean_reward_true}")
    _, dro_reward_true = true_evaluation_dro(
        seed,
        policy=policy,
        action_costs=action_costs,
        no_samples=1000000,
        delta=delta,
    )
    print(f"DRO mean reward from true evaluation: {dro_reward_true}")

    mean_rewards_seeds = {}
    # Run the simulations for different sample sizes
    for n_samples in n_samples_arr:
        print(f"Running simulations for n_samples: {n_samples}")
        mean_rewards_method = {
            "model": [],
            "data": [],
            "true": [],
            "dro_model": [],
            "dro_data": [],
            "dro_true": [],
        }
        for i in tqdm(range(no_sims), desc="Simulations"):
            (
                mean_reward_model,
                mean_dro_reward_model,
                mean_reward_data,
                mean_dro_reward_data,
                mean_reward_true,
                mean_dro_reward_true,
            ) = run_single(
                seed, policy, action_costs, n_samples, generation_seed=i * 10, delta=delta
            )
            mean_rewards_method["model"].append(float(mean_reward_model))
            mean_rewards_method["data"].append(float(mean_reward_data))
            mean_rewards_method["true"].append(float(mean_reward_true))
            mean_rewards_method["dro_model"].append(float(mean_dro_reward_model))
            mean_rewards_method["dro_data"].append(float(mean_dro_reward_data))
            mean_rewards_method["dro_true"].append(float(mean_dro_reward_true))

        mean_rewards_seeds[n_samples] = mean_rewards_method
    results_dict = {
        "results": mean_rewards_seeds,
        "weights": weights.tolist(),
        "bias": bias,
        "action_costs": action_costs,
        "true_reward": mean_reward_true,
        "true_dro_reward": dro_reward_true,
        "parameter_seed": seed,
        "generation_seed": generation_seed,
        "delta": delta,
    }
    # Save the results to a json file
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {results_file}")
    # Plot the results
    plot_results(results_dict, results_dir)


def plot_results(results_dict, results_dir):
    """Plot the results of the simulations.

    Args:
        results_dict (dict): Dictionary containing the results of the simulations.
    """
    parameter_seed = results_dict["parameter_seed"]
    generation_seed = results_dict["generation_seed"]
    delta = results_dict["delta"]
    # Create a directory for the plots
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # For each method, compute the mean and standard deviation for each sample size
    means_model = []
    stds_model = []
    means_data = []
    stds_data = []
    means_true = []
    stds_true = []
    true_mean_reward = results_dict["true_reward"]
    mean_reward_seed = results_dict["results"]
    n_samples_arr = list(mean_reward_seed.keys())
    for n_samples in n_samples_arr:
        mean_rewards = mean_reward_seed[n_samples]
        means_model.append(np.mean(mean_rewards["model"]))
        stds_model.append(np.std(mean_rewards["model"]))
        means_data.append(np.mean(mean_rewards["data"]))
        stds_data.append(np.std(mean_rewards["data"]))
        means_true.append(np.mean(mean_rewards["true"]))
        stds_true.append(np.std(mean_rewards["true"]))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples_arr, means_model, label="Model Evaluation", marker="o")
    plt.fill_between(
        n_samples_arr,
        np.array(means_model) - np.array(stds_model),
        np.array(means_model) + np.array(stds_model),
        alpha=0.2,
    )
    plt.plot(n_samples_arr, means_data, label="Data Evaluation", marker="o")
    plt.fill_between(
        n_samples_arr,
        np.array(means_data) - np.array(stds_data),
        np.array(means_data) + np.array(stds_data),
        alpha=0.2,
    )
    plt.plot(n_samples_arr, means_true, label="True Evaluation", marker="o")
    plt.fill_between(
        n_samples_arr,
        np.array(means_true) - np.array(stds_true),
        np.array(means_true) + np.array(stds_true),
        alpha=0.2,
    )
    plt.axhline(
        y=true_mean_reward,
        color="r",
        linestyle="--",
        label="True Mean Reward",
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward vs Number of Samples")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "mean_reward_vs_samples.png"))
    plt.show()

    # Compute the mean errors for each method
    errs_model = []
    errs_data = []
    errs_true = []
    for n_samples in n_samples_arr:
        mean_rewards = mean_reward_seed[n_samples]
        mean_err_model = np.mean(
            np.abs(np.array(mean_rewards["model"]) - true_mean_reward)
        )
        mean_err_data = np.mean(
            np.abs(np.array(mean_rewards["data"]) - true_mean_reward)
        )
        mean_err_true = np.mean(
            np.abs(np.array(mean_rewards["true"]) - true_mean_reward)
        )
        errs_model.append(mean_err_model)
        errs_data.append(mean_err_data)
        errs_true.append(mean_err_true)

    # Plot the error convergence
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples_arr, errs_model, label="Model Evaluation", marker="o")
    plt.plot(n_samples_arr, errs_data, label="Data Evaluation", marker="o")
    plt.plot(n_samples_arr, errs_true, label="True Evaluation", marker="o")
    plt.title("Error Convergence")
    plt.xlabel("Number of Samples")
    plt.ylabel("Absolute Error")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "error_convergence.png"))
    plt.show()

    # Repeat for DRO
    dro_means_model = []
    dro_stds_model = []
    dro_means_data = []
    dro_stds_data = []
    dro_means_true = []
    dro_stds_true = []
    true_dro_reward = results_dict["true_dro_reward"]
    dro_mean_reward_seed = results_dict["results"]
    dro_n_samples_arr = list(dro_mean_reward_seed.keys())
    for n_samples in dro_n_samples_arr:
        mean_rewards = dro_mean_reward_seed[n_samples]
        dro_means_model.append(np.mean(mean_rewards["dro_model"]))
        dro_stds_model.append(np.std(mean_rewards["dro_model"]))
        dro_means_data.append(np.mean(mean_rewards["dro_data"]))
        dro_stds_data.append(np.std(mean_rewards["dro_data"]))
        dro_means_true.append(np.mean(mean_rewards["dro_true"]))
        dro_stds_true.append(np.std(mean_rewards["dro_true"]))
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        dro_n_samples_arr,
        dro_means_model,
        label="Model Evaluation",
        marker="o",
    )
    plt.fill_between(
        dro_n_samples_arr,
        np.array(dro_means_model) - np.array(dro_stds_model),
        np.array(dro_means_model) + np.array(dro_stds_model),
        alpha=0.2,
    )
    plt.plot(
        dro_n_samples_arr,
        dro_means_data,
        label="Data Evaluation",
        marker="o",
    )
    plt.fill_between(
        dro_n_samples_arr,
        np.array(dro_means_data) - np.array(dro_stds_data),
        np.array(dro_means_data) + np.array(dro_stds_data),
        alpha=0.2,
    )
    plt.plot(
        dro_n_samples_arr,
        dro_means_true,
        label="True Evaluation",
        marker="o",
    )
    plt.fill_between(
        dro_n_samples_arr,
        np.array(dro_means_true) - np.array(dro_stds_true),
        np.array(dro_means_true) + np.array(dro_stds_true),
        alpha=0.2,
    )
    plt.axhline(
        y=true_dro_reward,
        color="r",
        linestyle="--",
        label="True DRO Reward",
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean DRO Reward")
    plt.title("Mean DRO Reward vs Number of Samples")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "mean_dro_reward_vs_samples.png"))
    plt.show()

    # Compute the mean errors for each method
    dro_errs_model = []
    dro_errs_data = []
    dro_errs_true = []
    for n_samples in dro_n_samples_arr:
        mean_rewards = dro_mean_reward_seed[n_samples]
        mean_err_model = np.mean(
            np.abs(np.array(mean_rewards["dro_model"]) - true_dro_reward)
        )
        mean_err_data = np.mean(
            np.abs(np.array(mean_rewards["dro_data"]) - true_dro_reward)
        )
        mean_err_true = np.mean(
            np.abs(np.array(mean_rewards["dro_true"]) - true_dro_reward)
        )
        dro_errs_model.append(mean_err_model)
        dro_errs_data.append(mean_err_data)
        dro_errs_true.append(mean_err_true)
    # Plot the error convergence
    plt.figure(figsize=(10, 6))
    plt.plot(
        dro_n_samples_arr,
        dro_errs_model,
        label="Model Evaluation",
        marker="o",
    )
    plt.plot(
        dro_n_samples_arr,
        dro_errs_data,
        label="Data Evaluation",
        marker="o",
    )
    plt.plot(
        dro_n_samples_arr,
        dro_errs_true,
        label="True Evaluation",
        marker="o",
    )
    plt.title("DRO Error Convergence")
    plt.xlabel("Number of Samples")
    plt.ylabel("Absolute Error")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "dro_error_convergence.png"))
    plt.show()


def run_single(parameter_seed, policy, action_costs, n_samples, generation_seed, delta=0.1):
    """Run a single simulation with the given parameters.

    Args:
        parameter_seed (int): Random seed for reproducibility.
        n_samples (int): Number of samples to generate.
        generation_seed (int): Seed for data generation.
    """
    X, A_one_hot, labels, gen_dict = generate_data(
        parameter_seed=parameter_seed,
        n_samples=n_samples,
        verbose=False,
        generation_seed=generation_seed,
    )

    # Get the weights and biases
    weights = gen_dict["weights"]
    bias = gen_dict["bias"]

    # Evaluate the policy using model evaluation
    actions_one_hot = apply_policy(X, policy)

    mean_reward_model, mean_dro_reward_model = evaluate_policy_model_dro(
        X, A_one_hot, labels, action_costs, actions_one_hot, delta=delta
    )

    # Evaluate the policy using data evaluation
    mean_reward_data, mean_dro_reward_data = evaluate_policy_data_dro(
        X, A_one_hot, labels, action_costs, actions_one_hot, delta=delta
    )

    # Evaluate the policy using true evaluation
    mean_reward_true, mean_dro_reward_true = true_evaluation_data_dro(
        weights, bias, X, action_costs, actions_one_hot, delta=delta
    )

    return (
        mean_reward_model,
        mean_dro_reward_model,
        mean_reward_data,
        mean_dro_reward_data,
        mean_reward_true,
        mean_dro_reward_true,
    )


if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 9
    generation_seed = 0
    delta = 0.1

    # Run the main function
    main(seed, generation_seed, delta)
