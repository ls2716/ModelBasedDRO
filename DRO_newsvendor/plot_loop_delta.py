"""Plot the results from the loop delta experiment."""
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Increase the font size for better readability
plt.rcParams.update({"font.size": 16})

def plot_results(results, delta_arr, delta_to_plot):
    """
    Plot the results of the KL-DRO experiment.

    Args:
        results (dict): Dictionary containing optimal order quantities and dro rewards.
        delta_arr (list): List of delta values.
        delta_to_plot (float): Delta value to plot.
    """
    # Create the plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")
    # Extract optimal order quantities and dro rewards
    optimal_order_quantities = results["optimal_order_quantities"].values()
    dro_rewards = results["dro_rewards"]

    # Plot the optimal order quantities
    plt.figure(figsize=(10, 5))
    plt.plot(delta_arr, optimal_order_quantities, marker="o")
    plt.title("Optimal Order Quantities vs Delta")
    plt.xlabel("Delta")
    plt.ylabel("Optimal Order Quantity")
    plt.grid()
    plt.savefig("plots/optimal_order_quantities.png")
    plt.close()

    # Plot the dro rewards
    plt.figure(figsize=(9, 5))
    # Plot the rewards for the standard order quantity
    dro_rewards_standard = [dro_rewards[str(delta)]["0"] for delta in delta_arr]
    plt.plot(delta_arr, dro_rewards_standard, marker="o", label="Standard (non-robust)")
    for delta_opt in delta_to_plot:
        dro_rewards_delta_opt = [
            dro_rewards[str(delta)]["{}".format(delta_opt)] for delta in delta_arr
        ]
        plt.plot(delta_arr, dro_rewards_delta_opt, marker="o", label="$\\delta_{train}=$" + f"{delta_opt:.3f}")

    # plt.title("$V^{DRO,\\delta}$ vs $\\delta$")
    plt.xlabel("$\\delta_{eval}$")
    plt.ylabel("Objective $V^{DRO}_{\\delta_{eval}}$")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Major grid lines should be solid and darker grey
    plt.grid(True, which='major', linestyle='-', linewidth=1, color='grey')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig("plots/dro_rewards.png")
    plt.close()


if __name__ == "__main__":
    # Load the results from the json file
    with open("results/experiment_results.json", "r") as f:
        results = json.load(f)

    # Define delta values to evaluate
    delta_arr = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    delta_to_plot = [0.005, 0.05, 0.1, 0.2]

    # Plot the results
    plot_results(results, delta_arr, delta_to_plot)