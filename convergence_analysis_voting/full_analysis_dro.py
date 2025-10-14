from generate_data import generate_data
from convergence_analysis_voting.evaluation_true import optimise_policy
from policy import NeuralNetworkPolicy
from convergence_analysis_voting.evaluation_DRO import (
    apply_policy,
    evaluate_policy_model_dro,
    evaluate_policy_data_dro,
)
from convergence_analysis_voting.evaluation_true import (
    true_evaluation,
    true_evaluation_data_dro,
    true_evaluation_dro,
)

import numpy as np
import os
import torch
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
    seed = 1
    generation_seed = 0
    delta = 0.1

    # Run the main function
    main(seed, generation_seed, delta)
