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
    with open(file_path, 'r') as f:
        result_dict = json.load(f)
    return result_dict


if __name__ == "__main__":

    # Increase the size of the font
    plt.rcParams.update({'font.size': 18})
    
    result_path = "results_dro"

    seeds = [0,1,2,3,4,5,6,7,8,9]

    # Initialize a dictionary to store the results
    seed_dict = {}

    for seed in seeds:
        # Construct the file path
        file_path = os.path.join(result_path, f"seed_{seed}_0_0.1", "results.json")
        
        # Load the result dictionary
        result_dict = load_result_dict(file_path)
        
        # Store the result dictionary in the seed_dict
        seed_dict[seed] = result_dict

    errors_dict = {}
    for seed in seeds:
        errors_dict[seed] = {}
        true_reward = seed_dict[seed]["true_reward"]
        true_dro_reward = seed_dict[seed]["true_dro_reward"]
        n_samples = seed_dict[seed]["results"].keys()
        for n in n_samples:
            model_rewards = seed_dict[seed]["results"][n]["model"]
            data_rewards = seed_dict[seed]["results"][n]["data"]
            true_rewards = seed_dict[seed]["results"][n]["true"]
            model_dro_rewards = seed_dict[seed]["results"][n]["dro_model"]
            data_dro_rewards = seed_dict[seed]["results"][n]["dro_data"]
            true_dro_rewards = seed_dict[seed]["results"][n]["dro_true"]
            errors_dict[seed][n] = {
                "model": np.abs(np.array(model_rewards) - true_reward),
                "data": np.abs(np.array(data_rewards) - true_reward),
                "true": np.abs(np.array(true_rewards) - true_reward),
                "model_dro": np.abs(np.array(model_dro_rewards) - true_dro_reward),
                "data_dro": np.abs(np.array(data_dro_rewards) - true_dro_reward),
                "true_dro": np.abs(np.array(true_dro_rewards) - true_dro_reward),
            }
    
    mean_error_model = []
    mean_error_data = []
    mean_error_true = []
    for n in n_samples:
        model_err_sum = 0
        data_err_sum = 0
        true_err_sum = 0
        for seed in seeds:
            model_err_sum += np.mean(errors_dict[seed][n]["model"])
            data_err_sum += np.mean(errors_dict[seed][n]["data"])
            true_err_sum += np.mean(errors_dict[seed][n]["true"])
        mean_error_model.append(model_err_sum / len(seeds))
        mean_error_data.append(data_err_sum / len(seeds))
        mean_error_true.append(true_err_sum / len(seeds))
    
    # Plot the errors
    n_samples_arr = np.array(list(map(int, n_samples)))
    plt.figure(figsize=(8, 6))
    plt.semilogx(n_samples_arr, mean_error_data, label="Data-based evaluation error")
    plt.semilogx(n_samples_arr, mean_error_model, label="Parametric evaluation error")
    plt.semilogx(n_samples_arr, mean_error_true, label="True data-based evaluation error")
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean absolute error")
    plt.title("Error convergence for standard objective")
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig("results_dro/error_convergence.png")
    plt.show()
    # exit(0)


    dro_error_model = []
    dro_error_data = []
    dro_error_true = []
    for n in n_samples:
        print(f"n_samples: {n}")
        model_err_sum = 0
        data_err_sum = 0
        true_err_sum = 0
        for seed in seeds:
            model_err_sum += np.mean(errors_dict[seed][n]["model_dro"])
            data_err_sum += np.mean(errors_dict[seed][n]["data_dro"])
            true_err_sum += np.mean(errors_dict[seed][n]["true_dro"])
        dro_error_model.append(model_err_sum / len(seeds))
        dro_error_data.append(data_err_sum / len(seeds))
        dro_error_true.append(true_err_sum / len(seeds))
    
    # Plot the errors
    plt.figure(figsize=(8, 6))
    plt.semilogx(n_samples_arr, dro_error_data, label="Data-based evaluation error")
    plt.semilogx(n_samples_arr, dro_error_model, label="Parametric evaluation error")
    plt.semilogx(n_samples_arr, dro_error_true, label="True data-based evaluation error")
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean absolute error")
    plt.title("Error convergence for DRO objective")
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig("results_dro/dro_error_convergence.png")
    plt.show()



