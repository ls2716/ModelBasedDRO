import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(file_path):
    """
    Load the results from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Loaded results.
    """
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    return results


def plot_errors(num_samples_arr, mean_dro_error_data, mean_dro_error_mb, beta):
    """
    Plot the errors for the different number of samples.
    Args:
        num_samples_arr (list): List of number of samples.
        mean_dro_error_data (list): List of mean errors for the data-driven approach.
        mean_dro_error_mb (list): List of mean errors for the model-based approach.
        beta (float): Beta value for the experiment.
    """
    # Plot on a normal scale
    plt.figure(figsize=(8, 6))
    plt.plot(num_samples_arr, mean_dro_error_data, label="Data-based evaluation", marker="o")
    plt.plot(num_samples_arr, mean_dro_error_mb, label="Parametric evaluation", marker="o")
    plt.xlabel("Number of samples")
    plt.ylabel("Mean $l_\\infty$ error")
    plt.title(f"$l_\\infty$ error in DRO evaluation for \\beta = {beta}")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/mb_vs_mf_error_{beta}.png")
    plt.close()

    # Plot on a log scale
    plt.figure(figsize=(8, 6))
    plt.plot(num_samples_arr, mean_dro_error_data, label="Data-based evaluation", marker="o")
    plt.plot(num_samples_arr, mean_dro_error_mb, label="Parametric evaluation", marker="o") 
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of samples")
    plt.ylabel("Mean $l_\\infty$ error")
    plt.title(f"$l_\\infty$ error in DRO evaluation for \\beta = {beta}")
    # Add grid lines with both major and minor ticks
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/mb_vs_mf_error_log_{beta}.png")
    plt.close()

if __name__ == "__main__":
    alpha = -15
    beta = 0.5
    no_points = 10
    delta = 0.1

    # Increase the size of the font
    plt.rcParams.update({"font.size": 18})

    # Path to the results
    result_path = "results"

    # Load the results
    results = load_results(os.path.join(result_path, f"results_{beta}.pkl"))

    # Extract the number of samples
    num_samples_arr = list(results.keys())

    mean_dro_error_data = []
    mean_dro_error_mb = []

    # Compute l_infty errors for the evaluation for each number of samples
    for num_samples in num_samples_arr:
        profits_dro = results[num_samples]["profits_dro"]
        mb_profits_action_space = results[num_samples]["mb_profits_dro_action_space"]
        true_profits_action_space = results[num_samples]["true_dro_profits_action_space"]
        # Convert to numpy arrays
        profits_dro = np.array(profits_dro)
        mb_profits_action_space = np.array(mb_profits_action_space)
        true_profits_action_space = np.array(true_profits_action_space)
        # Compute the errors
        error_dro_data = np.abs(profits_dro - true_profits_action_space)
        error_dro_mb = np.abs(mb_profits_action_space - true_profits_action_space)

        # Compute the l_infty errors (axis=1)
        error_dro_data = np.max(error_dro_data, axis=1)
        error_dro_mb = np.max(error_dro_mb, axis=1)
        # # Compute the mean absolute error (axis=1)
        # error_dro_data = np.mean(error_dro_data, axis=1)
        # error_dro_mb = np.mean(error_dro_mb, axis=1)

        # Compute the mean error  (axis=0)
        mean_error_dro_data = np.mean(error_dro_data)
        mean_error_dro_mb = np.mean(error_dro_mb)

        # Append the mean errors to the lists
        mean_dro_error_data.append(mean_error_dro_data)
        mean_dro_error_mb.append(mean_error_dro_mb)
    
    # Convert to numpy arrays
    mean_dro_error_data = np.array(mean_dro_error_data)
    mean_dro_error_mb = np.array(mean_dro_error_mb)

    # Plot the results
    plot_errors(num_samples_arr, mean_dro_error_data, mean_dro_error_mb, beta)
