import matplotlib.pyplot as plt
import numpy as np


import json
import os


def load_result_dict(file_path):
    """
    Load the result dictionary from a file.

    Args:
        file_path (str): Path to the file containing the result dictionary.

    Returns:
        dict: Loaded result dictionary.
    """
    with open(file_path, "r") as f:
        result_dict = json.load(f)
    return result_dict


def plot_mean_convergence(
    n_samples_arr,
    mean_model_rewards,
    mean_data_rewards,
    mean_true_rewards,
    std_model_rewards,
    std_data_rewards,
    std_true_rewards,
    true_reward,
    file_path=None,
    seed=None,
):
    """
    Plot the mean convergence of rewards.

    Args:
        n_samples_arr (np.ndarray): Array of sample sizes.
        mean_model_rewards (list): List of mean model rewards.
        mean_data_rewards (list): List of mean data rewards.
        mean_true_rewards (list): List of mean true rewards.
        std_model_rewards (list): List of standard deviations for model rewards.
        std_data_rewards (list): List of standard deviations for data rewards.
        std_true_rewards (list): List of standard deviations for true rewards.
    """
    if seed is None:
        raise ValueError("Seed must be provided for convergence plot.")
    plt.figure(figsize=(8, 6))
    plt.plot(n_samples_arr, mean_model_rewards, label="Parametric evaluation")
    plt.fill_between(
        n_samples_arr,
        np.array(mean_model_rewards) - np.array(std_model_rewards),
        np.array(mean_model_rewards) + np.array(std_model_rewards),
        alpha=0.2,
    )
    plt.plot(n_samples_arr, mean_data_rewards, label="Data-based evaluation")
    plt.fill_between(
        n_samples_arr,
        np.array(mean_data_rewards) - np.array(std_data_rewards),
        np.array(mean_data_rewards) + np.array(std_data_rewards),
        alpha=0.2,
    )
    plt.plot(n_samples_arr, mean_true_rewards, label="True data-based evaluation")
    plt.fill_between(
        n_samples_arr,
        np.array(mean_true_rewards) - np.array(std_true_rewards),
        np.array(mean_true_rewards) + np.array(std_true_rewards),
        alpha=0.2,
    )
    plt.axhline(
        y=true_reward, color="r", linestyle="--", label="True standard objective"
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Reward")
    plt.title("Convergence of standard objective")
    plt.legend()
    plt.xscale("log")
    plt.minorticks_on()
    plt.grid()
    plt.tight_layout()
    if file_path is not None:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(os.path.join(file_path, f"mean_convergence_seed_{seed}.png"))
    # plt.show()
    plt.close()


def plot_mean_dro_convergence(
    n_samples_arr,
    mean_model_dro_rewards,
    mean_data_dro_rewards,
    mean_true_dro_rewards,
    std_model_dro_rewards,
    std_data_dro_rewards,
    std_true_dro_rewards,
    true_dro_reward,
    file_path=None,
    seed=None,
):
    """
    Plot the mean convergence of DRO rewards.

    Args:
        n_samples_arr (np.ndarray): Array of sample sizes.
        mean_model_dro_rewards (list): List of mean model DRO rewards.
        mean_data_dro_rewards (list): List of mean data DRO rewards.
        mean_true_dro_rewards (list): List of mean true DRO rewards.
        std_model_dro_rewards (list): List of standard deviations for model DRO rewards.
        std_data_dro_rewards (list): List of standard deviations for data DRO rewards.
        std_true_dro_rewards (list): List of standard deviations for true DRO rewards.
    """
    if seed is None:
        raise ValueError("Seed must be provided for DRO convergence plot.")
    plt.figure(figsize=(8, 6))
    plt.plot(n_samples_arr, mean_model_dro_rewards, label="Parametric evaluation")
    plt.fill_between(
        n_samples_arr,
        np.array(mean_model_dro_rewards) - np.array(std_model_dro_rewards),
        np.array(mean_model_dro_rewards) + np.array(std_model_dro_rewards),
        alpha=0.2,
    )
    plt.plot(n_samples_arr, mean_data_dro_rewards, label="Data-based evaluation")
    plt.fill_between(
        n_samples_arr,
        np.array(mean_data_dro_rewards) - np.array(std_data_dro_rewards),
        np.array(mean_data_dro_rewards) + np.array(std_data_dro_rewards),
        alpha=0.2,
    )
    plt.plot(n_samples_arr, mean_true_dro_rewards, label="True data-based evaluation")
    plt.fill_between(
        n_samples_arr,
        np.array(mean_true_dro_rewards) - np.array(std_true_dro_rewards),
        np.array(mean_true_dro_rewards) + np.array(std_true_dro_rewards),
        alpha=0.2,
    )
    plt.axhline(
        y=true_dro_reward, color="r", linestyle="--", label="True DRO objective"
    )
    plt.xlabel("Number of samples")
    plt.ylabel("Mean reward")
    plt.title("Convergence of DRO objective")
    plt.legend()
    plt.xscale("log")
    plt.minorticks_on()
    plt.grid()
    plt.tight_layout()
    if file_path is not None:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(os.path.join(file_path, f"mean_dro_convergence_seed_{seed}.png"))
    plt.close()


if __name__ == "__main__":
    # Increase the size of the font
    plt.rcParams.update({"font.size": 18})

    result_path = "results_dro"

    seeds = [1,2,3]

    # Initialize a dictionary to store the results
    seed_dict = {}

    for seed in seeds:
        # Construct the file path
        file_path = os.path.join(result_path, f"seed_{seed}_0_0.1", "results.json")

        # Load the result dictionary
        result_dict = load_result_dict(file_path)

        # Store the result dictionary in the seed_dict
        seed_dict[seed] = result_dict

    results_dict = {}
    for seed in seeds:
        results_dict[seed] = {}
        true_reward = seed_dict[seed]["true_reward"]
        true_dro_reward = seed_dict[seed]["true_dro_reward"]
        n_samples = seed_dict[seed]["results"].keys()
        mean_model_rewards = []
        mean_data_rewards = []
        mean_true_rewards = []
        mean_model_dro_rewards = []
        mean_data_dro_rewards = []
        mean_true_dro_rewards = []
        std_model_rewards = []
        std_data_rewards = []
        std_true_rewards = []
        std_model_dro_rewards = []
        std_data_dro_rewards = []
        std_true_dro_rewards = []
        for n in n_samples:
            model_rewards = seed_dict[seed]["results"][n]["model"]
            data_rewards = seed_dict[seed]["results"][n]["data"]
            true_rewards = seed_dict[seed]["results"][n]["true"]
            model_dro_rewards = seed_dict[seed]["results"][n]["dro_model"]
            data_dro_rewards = seed_dict[seed]["results"][n]["dro_data"]
            true_dro_rewards = seed_dict[seed]["results"][n]["dro_true"]
            mean_model_rewards.append(np.mean(model_rewards))
            mean_data_rewards.append(np.mean(data_rewards))
            mean_true_rewards.append(np.mean(true_rewards))
            mean_model_dro_rewards.append(np.mean(model_dro_rewards))
            mean_data_dro_rewards.append(np.mean(data_dro_rewards))
            mean_true_dro_rewards.append(np.mean(true_dro_rewards))
            std_model_rewards.append(np.std(model_rewards))
            std_data_rewards.append(np.std(data_rewards))
            std_true_rewards.append(np.std(true_rewards))
            std_model_dro_rewards.append(np.std(model_dro_rewards))
            std_data_dro_rewards.append(np.std(data_dro_rewards))
            std_true_dro_rewards.append(np.std(true_dro_rewards))
        n_samples_arr = np.array(list(n_samples), dtype=int)

        file_path = os.path.join(result_path, f"seed_{seed}_0_0.1", "plots")
        # Plot the mean convergence
        plot_mean_convergence(
            n_samples_arr,
            mean_model_rewards,
            mean_data_rewards,
            mean_true_rewards,
            std_model_rewards,
            std_data_rewards,
            std_true_rewards,
            true_reward,
            file_path=file_path,
            seed=seed,
        )
        plot_mean_dro_convergence(
            n_samples_arr,
            mean_model_dro_rewards,
            mean_data_dro_rewards,
            mean_true_dro_rewards,
            std_model_dro_rewards,
            std_data_dro_rewards,
            std_true_dro_rewards,
            true_dro_reward,
            file_path=file_path,
            seed=seed,
        )
