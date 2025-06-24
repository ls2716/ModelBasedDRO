import numpy as np
import os
import matplotlib.pyplot as plt

def plot_results(results, plot_path=None):
    """Plot the results of the experiment."""
    # Extract the results from the dictionary
    mb_profits = [result["mb"][1] for result in results]
    bayesian_profits = [result["bayesian"][1] for result in results]
    robust_bayesian_profits = {}
    robust_mb_profits = {}
    for delta_mb in results[0]["robust_mb"].keys():
        robust_mb_profits[delta_mb] = [
            result["robust_mb"][delta_mb][1] for result in results
        ]
    bayesian_deltas = results[0]["robust_bayesian"].keys()
    for delta_bayesian in bayesian_deltas:
        robust_bayesian_profits[delta_bayesian] = [
            result["robust_bayesian"][delta_bayesian][1] for result in results
        ]

    # Compute the mean profits for each method
    mb_mean_profit = np.mean(mb_profits)
    bayesian_mean_profit = np.mean(bayesian_profits)
    robust_bayesian_mean_profits = {
        delta_bayesian: np.mean(profits)
        for delta_bayesian, profits in robust_bayesian_profits.items()
    }
    robust_mb_mean_profits = {
        delta_mb: np.mean(profits) for delta_mb, profits in robust_mb_profits.items()
    }

    # Compute the standard deviation of the profits for each method
    mb_std_profit = np.std(mb_profits)
    bayesian_std_profit = np.std(bayesian_profits)
    robust_bayesian_std_profits = {
        delta_bayesian: np.std(profits)
        for delta_bayesian, profits in robust_bayesian_profits.items()
    }
    robust_mb_std_profits = {
        delta_mb: np.std(profits) for delta_mb, profits in robust_mb_profits.items()
    }

    # # Plot the profits as lines vs seed
    # plt.figure(figsize=(10, 6))
    # plt.plot(mb_profits, label="MB", marker="x")
    # plt.plot(bayesian_profits, label="Bayesian", marker="o")
    # for delta_bayesian, profits in robust_bayesian_profits.items():
    #     plt.plot(profits, label=f"Robust Bayesian (delta={delta_bayesian})", marker="o")
    # for delta_mb, profits in robust_mb_profits.items():
    #     plt.plot(profits, label=f"Robust MB (delta={delta_mb})", marker="x")
    # plt.xlabel("Simulation")
    # plt.ylabel("Profit")
    # plt.title("Profit vs Simulation")
    # plt.legend()
    # plt.grid()
    # if plot_path:
    #     plt.savefig(plot_path+"PvsSim.png")
    # else:
    #     plt.savefig("profit_vs_simulation.png")
    # plt.show()

    # Plot the mean profit vs std profit for each method on a point plot
    plt.figure(figsize=(10, 6))

    # Initialize MB lists
    mb_stds = [mb_std_profit]
    mb_means = [mb_mean_profit]
    mb_deltas = [0]  # Non-robust MB is delta=0

    # Initialize Bayesian lists
    bayesian_stds = [bayesian_std_profit]
    bayesian_means = [bayesian_mean_profit]
    bayesian_deltas = [0]  # Non-robust Bayesian is delta=0

    # Add robust MB points
    for delta_mb in robust_mb_mean_profits.keys():
        # if float(delta_mb) > 0.1:
        #     continue
        mb_stds.append(robust_mb_std_profits[delta_mb])
        mb_means.append(robust_mb_mean_profits[delta_mb])
        mb_deltas.append(delta_mb)

    # Add robust Bayesian points
    for delta_bayesian in robust_bayesian_mean_profits.keys():
        bayesian_stds.append(robust_bayesian_std_profits[delta_bayesian])
        bayesian_means.append(robust_bayesian_mean_profits[delta_bayesian])
        bayesian_deltas.append(delta_bayesian)

    # # Sort by delta for plotting lines
    # sorted_mb = sorted(zip(mb_deltas, mb_stds, mb_means))
    # sorted_bayesian = sorted(zip(bayesian_deltas, bayesian_stds, bayesian_means))

    # mb_deltas, mb_stds, mb_means = zip(*sorted_mb)
    # bayesian_deltas, bayesian_stds, bayesian_means = zip(*sorted_bayesian)

    # Plot MB method
    plt.plot(mb_stds, mb_means, color="tab:blue", marker="x", label="MB Method")
    for x, y, delta in zip(mb_stds, mb_means, mb_deltas):
        plt.annotate(
            f"$\delta={delta}$", 
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='left',
            fontsize=8,
            color="tab:blue"
        )

    # Plot Bayesian method
    plt.plot(bayesian_stds, bayesian_means, color="tab:orange", marker="o", label="Bayesian Method")
    for x, y, delta in zip(bayesian_stds, bayesian_means, bayesian_deltas):
        plt.annotate(
            f"$\delta={delta}$", 
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='left',
            fontsize=8,
            color="tab:orange"
        )

    plt.ylabel("Mean Profit")
    plt.xlabel("Standard Deviation of Profit")
    plt.title("Mean vs Standard Deviation of Profit")
    plt.legend()
    plt.grid()
    if plot_path:
        plt.savefig(plot_path+"mean_vs_std_profit.png")
    else:
        plt.savefig("mean_vs_std_profit.png")
    plt.show()



if __name__ == "__main__":

    # Read output file from arguments
    import argparse
    import json
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
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Extract the data
    metadata = data["metadata"]
    results = data["results"]

    # Extract the parameters
    num_samples = metadata["num_samples"]
    alpha = metadata["alpha"]
    beta = metadata["beta"]
    price_range_multiplier = metadata["price_range_multiplier"]

    plot_name = f"exp_{alpha}_{beta}_{num_samples}_{price_range_multiplier}"
    plot_path = os.path.join("images", plot_name)
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Plot the results
    plot_results(results, plot_path=plot_path)
    print(f"Plot saved to {plot_path}")
