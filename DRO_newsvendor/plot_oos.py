import matplotlib.pyplot as plt
import numpy as np

def plot_results(data_rewards, model_rewards, naive_data_rewards, naive_model_rewards, suffix):


    plt.plot(
        data_rewards,
        label="Data-based KL-DRO",
        marker="o",
    )
    plt.plot(
        model_rewards,
        label="Model-based KL-DRO",
        marker="o",
    )
    plt.plot(
        naive_data_rewards,
        label="Naive data",
        marker="o",
    )
    plt.plot(
        naive_model_rewards,
        label="Naive model",
        marker="o",
    )
    plt.title("Out-of-sample evaluation of KL-DRO")
    plt.xlabel("Experiment")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/oos_evaluation_{suffix}.png")
    plt.show()

    # Plot data rewards vs model rewards
    plt.scatter(data_rewards, model_rewards)
    plt.plot(
        [min(data_rewards), max(data_rewards)],
        [min(data_rewards), max(data_rewards)],
        linestyle="--",
        color="gray",
    )
    plt.title("Data-based KL-DRO vs Model-based KL-DRO")
    plt.xlabel("Data-based KL-DRO Reward")
    plt.ylabel("Model-based KL-DRO Reward")
    plt.grid()
    plt.savefig(f"plots/oos_evaluation_scatter_data_model_{suffix}.png")
    plt.show()



if __name__ == "__main__":
    import json
    import os

    delta = 0.01
    no_samples = 50

    # Load the results from the json file
    results_folder = "results"
    results_file = os.path.join(
        results_folder,
        f"results_delta_{delta}_no_samples_{no_samples}.json",
    )
    with open(results_file, "r") as f:
        results = json.load(f)

    data_rewards = results["data_rewards"]
    model_rewards = results["model_rewards"]
    naive_data_rewards = results["naive_data_rewards"]
    naive_model_rewards = results["naive_model_rewards"]
    data_var_rewards = results["data_var_rewards"]
    model_var_rewards = results["model_var_rewards"]
    naive_data_var_rewards = results["naive_data_var_rewards"]
    naive_model_var_rewards = results["naive_model_var_rewards"]
    suffix = f"delta_{delta}_no_samples_{no_samples}"
    plot_results(
        data_rewards,
        model_rewards,
        naive_data_rewards,
        naive_model_rewards,
        suffix=suffix,
    )
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
    print("Standard deviations of the rewards:")
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
