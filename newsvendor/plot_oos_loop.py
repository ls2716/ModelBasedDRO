import matplotlib.pyplot as plt
import numpy as np

# Increase the font size of the plots
plt.rcParams.update({"font.size": 16})
# Set the fontsize of the text annotations




def plot_results(
    data_rewards, model_rewards, naive_data_rewards, naive_model_rewards, suffix
):
    plt.figure(figsize=(10, 6))
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
    plt.figure(figsize=(10, 6))
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


def get_mean_and_std(delta, no_samples):
    """
    Get the mean and std of the rewards for a given delta and number of samples.
    """
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
    data_std_rewards = results["data_var_rewards"]
    model_std_rewards = results["model_var_rewards"]
    naive_data_std_rewards = results["naive_data_var_rewards"]
    naive_model_std_rewards = results["naive_model_var_rewards"]

    # Compute the means
    data_mean = np.mean(data_rewards)
    model_mean = np.mean(model_rewards)
    naive_data_mean = np.mean(naive_data_rewards)
    naive_model_mean = np.mean(naive_model_rewards)
    # Compute the variance by adding the veriance of means and mean of variances
    data_std = np.sqrt(np.mean(np.square(data_std_rewards)) + np.var(data_rewards))
    model_std = np.sqrt(np.mean(np.square(model_std_rewards)) + np.var(model_rewards))
    naive_data_std = np.sqrt(
        np.mean(np.square(naive_data_std_rewards)) + np.var(naive_data_rewards)
    )
    naive_model_std = np.sqrt(
        np.mean(np.square(naive_model_std_rewards)) + np.var(naive_model_rewards)
    )

    # Print the mean and std of the rewards
    print(
        f"Data-based KL-DRO: Mean Reward: {data_mean}, "
        f"Std Reward: {data_std}"
    )
    print(
        f"Model-based KL-DRO: Mean Reward: {model_mean}, "
        f"Std Reward: {model_std}"
    )
    print(
        f"Naive data: Mean Reward: {naive_data_mean}, "
        f"Std Reward: {naive_data_std}"
    )
    print(
        f"Naive model: Mean Reward: {naive_model_mean}, "
        f"Std Reward: {naive_model_std}"
    )
    print('-------------------------------')

    return (
        data_mean,
        model_mean,
        naive_data_mean,
        naive_model_mean,
        data_std,
        model_std,
        naive_data_std,
        naive_model_std,
    )


if __name__ == "__main__":
    import json
    import os

    deltas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    no_samples = 10

    data_means = []
    model_means = []
    naive_data_means = []
    naive_model_means = []
    data_stds = []
    model_stds = []
    naive_data_stds = []
    naive_model_stds = []
    # Create a folder for the plots
    os.makedirs("plots", exist_ok=True)

    # Get the mean and std of the rewards for each delta
    for delta in deltas:
        (
            data_mean,
            model_mean,
            naive_data_mean,
            naive_model_mean,
            data_std,
            model_std,
            naive_data_std,
            naive_model_std,
        ) = get_mean_and_std(delta, no_samples)
        data_means.append(data_mean)
        model_means.append(model_mean)
        naive_data_means.append(naive_data_mean)
        naive_model_means.append(naive_model_mean)
        data_stds.append(data_std)
        model_stds.append(model_std)
        naive_data_stds.append(naive_data_std)
        naive_model_stds.append(naive_model_std)
    
    # Plot the means vs stds for each delta
    upper_y_limit = {
        10: -40.4,
        20: -39.5
        # 50: -160.0,
    }
    plt.figure(figsize=(12, 6))
    plt.plot(data_stds, data_means, label="Data-based KL-DRO", marker="x")
    plt.plot(model_stds, model_means, label="Parametric KL-DRO", marker="x")
    plt.scatter(
        naive_data_stds[0], naive_data_means[0], label="Data-based standard", marker="o"
    )
    plt.scatter(
        naive_model_stds[0], naive_model_means[0], label="Parametric standard", marker="o"
    )
    # Label each point with the delta value
    for i, delta in enumerate(deltas):
        plt.annotate(
            f"{delta:.3f}",
            (data_stds[i], data_means[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )
        plt.annotate(
            f"{delta:.3f}",
            (model_stds[i], model_means[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )
    if no_samples in upper_y_limit:
        plt.ylim(
            None,
            upper_y_limit[no_samples],
        )
    # Increase the upper limit of the y-axis by 0.5
    plt.title(f"Out-of-sample evaluation of KL-DRO for N={no_samples}")
    plt.xlabel("Standard deviation of the reward $\sqrt{\overline{\sigma^2}}$")
    plt.ylabel("Avergage mean reward $\overline{\mu}$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/oos_evaluation_means_vs_stds_no_samples_{no_samples}.png")
    plt.show()

