import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_results(results, plot_path=None):
    """Plot the results of the experiment.

    Args:
        results (dict): The results of the experiment.
    """
    result_mb = results["result_mb"]
    result_bayesian = results["result_bayesian"]
    result_robust_mb = results["result_robust_mb"]
    result_robust_bayesian = results["result_robust_bayesian"]

    # Plot the mean and std for the profit distributions
    mean_profit_mb = np.mean(result_mb[1])
    std_profit_mb = np.std(result_mb[1])
    mean_profit_bayesian = np.mean(result_bayesian[1])
    std_profit_bayesian = np.std(result_bayesian[1])

    plt.figure(figsize=(10, 6))
    plt.title("Profit Mean vs Std")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Mean Profit")

    # Initialize lists to store points for lines
    stds_mb = []
    means_mb = []
    delta_vals_mb = []

    stds_bayesian = []
    means_bayesian = []
    delta_vals_bayesian = []

    # Non-robust MB method (delta = 0)
    stds_mb.append(std_profit_mb)
    means_mb.append(mean_profit_mb)
    delta_vals_mb.append(0)

    # Non-robust Bayesian method (delta = 0)
    stds_bayesian.append(std_profit_bayesian)
    means_bayesian.append(mean_profit_bayesian)
    delta_vals_bayesian.append(0)

    # Robust MB methods
    for delta_mb, (
        optimised_robust_price_mb,
        profit_distr_robust_mb,
    ) in result_robust_mb.items():
        stds_mb.append(np.std(profit_distr_robust_mb))
        means_mb.append(np.mean(profit_distr_robust_mb))
        delta_vals_mb.append(delta_mb)

    # Robust Bayesian methods
    for delta_bayesian, (
        optimised_robust_price_bayesian,
        profit_distr_robust_bayesian,
    ) in result_robust_bayesian.items():
        stds_bayesian.append(np.std(profit_distr_robust_bayesian))
        means_bayesian.append(np.mean(profit_distr_robust_bayesian))
        delta_vals_bayesian.append(delta_bayesian)

    # Sort by delta to ensure the lines are connected nicely
    sorted_mb = sorted(zip(delta_vals_mb, stds_mb, means_mb))
    sorted_bayesian = sorted(
        zip(delta_vals_bayesian, stds_bayesian, means_bayesian)
    )

    delta_vals_mb, stds_mb, means_mb = zip(*sorted_mb)
    delta_vals_bayesian, stds_bayesian, means_bayesian = zip(*sorted_bayesian)

    # Plot MB method
    plt.plot(stds_mb, means_mb, color="tab:blue", marker="x", label="MB Method")
    for x, y, delta in zip(stds_mb, means_mb, delta_vals_mb):
        plt.annotate(
            f"$\delta={delta}$",
            (x, y),
            textcoords="offset points",
            xytext=(2, 15),
            ha="left",
            fontsize=8,
            color="tab:blue",
        )

    # Plot Bayesian method
    plt.plot(
        stds_bayesian,
        means_bayesian,
        color="tab:orange",
        marker="o",
        label="Bayesian Method",
    )
    for x, y, delta in zip(stds_bayesian, means_bayesian, delta_vals_bayesian):
        plt.annotate(
            f"$\delta={delta}$",
            (x, y),
            textcoords="offset points",
            xytext=(2, -15),
            ha="left",
            fontsize=8,
            color="tab:orange",
        )

    plt.legend()
    plt.grid()
    # plt.gca().set_aspect("equal")
    if plot_path:
        plt.savefig(plot_path + "_mean_std.png")
    # plt.show()

    # Plot the mean and percentile for the profit distributions
    percentile = 5
    mean_profit_mb = np.mean(result_mb[1])
    perc5_profit_mb = np.percentile(result_mb[1], percentile)

    mean_profit_bayesian = np.mean(result_bayesian[1])
    perc5_profit_bayesian = np.percentile(result_bayesian[1], percentile)

    plt.figure(figsize=(10, 6))
    plt.title(
        f"Profit Mean vs {percentile}% Percentile"
    )
    plt.xlabel(f"{percentile}% Percentile of Profit")
    plt.ylabel("Mean Profit")

    # Initialize lists to store points for lines
    perc5s_mb = []
    means_mb = []
    delta_vals_mb = []

    perc5s_bayesian = []
    means_bayesian = []
    delta_vals_bayesian = []

    # Non-robust MB method (delta = 0)
    perc5s_mb.append(perc5_profit_mb)
    means_mb.append(mean_profit_mb)
    delta_vals_mb.append(0)

    # Non-robust Bayesian method (delta = 0)
    perc5s_bayesian.append(perc5_profit_bayesian)
    means_bayesian.append(mean_profit_bayesian)
    delta_vals_bayesian.append(0)

    # Robust MB methods
    for delta_mb, (
        optimised_robust_price_mb,
        profit_distr_robust_mb,
    ) in result_robust_mb.items():
        perc5s_mb.append(np.percentile(profit_distr_robust_mb, percentile))
        means_mb.append(np.mean(profit_distr_robust_mb))
        delta_vals_mb.append(delta_mb)

    # Robust Bayesian methods
    for delta_bayesian, (
        optimised_robust_price_bayesian,
        profit_distr_robust_bayesian,
    ) in result_robust_bayesian.items():
        perc5s_bayesian.append(
            np.percentile(profit_distr_robust_bayesian, percentile)
        )
        means_bayesian.append(np.mean(profit_distr_robust_bayesian))
        delta_vals_bayesian.append(delta_bayesian)

    # Sort by delta to ensure the lines are connected nicely
    sorted_mb = sorted(zip(delta_vals_mb, perc5s_mb, means_mb))
    sorted_bayesian = sorted(
        zip(delta_vals_bayesian, perc5s_bayesian, means_bayesian)
    )

    delta_vals_mb, perc5s_mb, means_mb = zip(*sorted_mb)
    delta_vals_bayesian, perc5s_bayesian, means_bayesian = zip(*sorted_bayesian)

    # Plot MB method
    plt.plot(perc5s_mb, means_mb, color="tab:blue", marker="x", label="MB Method")
    for x, y, delta in zip(perc5s_mb, means_mb, delta_vals_mb):
        plt.annotate(
            f"$\delta={delta}$",
            (x, y),
            textcoords="offset points",
            xytext=(2, 15),
            ha="left",
            fontsize=8,
            color="tab:blue",
        )

    # Plot Bayesian method
    plt.plot(
        perc5s_bayesian,
        means_bayesian,
        color="tab:orange",
        marker="o",
        label="Bayesian Method",
    )
    for x, y, delta in zip(perc5s_bayesian, means_bayesian, delta_vals_bayesian):
        plt.annotate(
            f"$\delta={delta}$",
            (x, y),
            textcoords="offset points",
            xytext=(2, -15),
            ha="left",
            fontsize=8,
            color="tab:orange",
        )

    # Set aspect ratio to 1:1
    # plt.gca().set_aspect("equal")
    plt.legend()
    plt.grid()
    if plot_path:
        plt.savefig(plot_path + "_mean_percentile.png")
    # plt.show()

    # Plot the profit distributions for MB, Bayesian and robust methods with delta 0.2
    delta_to_plot = 0.05
    # Choose bins based on the min and max profit values for MB and Bayesian method
    min_profit = min(result_mb[1].min(), result_bayesian[1].min())
    max_profit = max(result_mb[1].max(), result_bayesian[1].max())
    no_bins = 80
    bins = list(np.linspace(min_profit, max_profit, num=no_bins + 1, endpoint=True))
    plt.figure(figsize=(10, 6))
    plt.title(
        f"Profit Distribution delta={delta_to_plot}"
    )
    plt.xlabel("Profit")
    plt.ylabel("Density")
    plt.hist(
        result_mb[1],
        bins=bins,
        density=True,
        alpha=0.5,
        label="Profit distribution MB",
    )
    plt.hist(
        result_bayesian[1],
        bins=bins,
        density=True,
        alpha=0.5,
        label="Profit distribution Bayesian",
    )
    if delta_to_plot in result_robust_mb:
        plt.hist(
            result_robust_mb[delta_to_plot][1],
            bins=bins,
            density=True,
            alpha=0.5,
            label=f"Profit distribution MB (delta={delta_to_plot})",
        )
    if delta_to_plot in result_robust_bayesian:
        plt.hist(
            result_robust_bayesian[delta_to_plot][1],
            bins=bins,
            density=True,
            alpha=0.5,
            label=f"Profit distribution Bayesian (delta={delta_to_plot})",
        )
    plt.legend()
    plt.grid()
    if plot_path:
        plt.savefig(plot_path + "_Pdist.png")
    # plt.show()


if __name__ == "__main__":

    # Read output file from arguments
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Plot results from a output file")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output file",
    )

    # Read the arguments
    args = parser.parse_args()
    output_file = args.output_file
    if output_file is None:
        raise ValueError("Output file not provided. Use --output_file to specify it.")
    if not os.path.exists(output_file):
        raise ValueError(f"Output file {output_file} does not exist.")
    
    # Read the output file
    with open(output_file, "rb") as f:
        data = pickle.load(f)
    
    # Extract the data
    metadata = data["metadata"]
    results = data["results"]

    # Extract the parameters
    alpha = metadata["alpha"]
    beta = metadata["beta"]
    beta_variance = metadata["beta_variance"]

    plot_name = f"exp_{alpha}_{beta}_{beta_variance}"
    plot_path = os.path.join("images", plot_name)
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Plot the results
    plot_results(results, plot_path=plot_path)
    print(f"Plot saved to {plot_path}")