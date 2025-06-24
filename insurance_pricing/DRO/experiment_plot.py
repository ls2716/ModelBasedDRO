"""Plot the results of the experiment."""

import json
import matplotlib.pyplot as plt

# Increase the default font size for better readability
plt.rcParams.update({'font.size': 16})


if __name__ == "__main__":

    # Set the seed
    seed = 0
    # Load the evaluation dictionary from the path
    evaluation_dict_path = f"seed_{seed}/evaluation_dict.json"
    with open(evaluation_dict_path, "r") as f:
        evaluation_dict = json.load(f)

    to_plot = [
        "standard",
        "robust_delta_0.01",
        "robust_delta_0.05",
        "robust_delta_0.1",
    ]
    labels = {
        "standard": "standard (non-robust)",
        "robust_delta_0.01": "robust $\\delta_{train}=0.01$",
        "robust_delta_0.05": "robust $\\delta_{train}=0.05$",
        "robust_delta_0.1": "robust $\\delta_{train}=0.1$",
    }

    # Plot the results
    plt.figure(figsize=(10, 6))
    for prices_type in to_plot:
        if prices_type not in evaluation_dict:
            print(f"Warning: {prices_type} not found in evaluation dictionary.")
            continue
        results_prices_type = evaluation_dict[prices_type]

        deltas = results_prices_type.keys()
        profits = []
        for delta in deltas:
            profits.append(results_prices_type[str(delta)])
        deltas = [float(delta) for delta in deltas]

        plt.plot(
            deltas,
            profits,
            marker="o",
            label=labels.get(prices_type, prices_type),
        )
    plt.xlabel("$\\delta_{eval}$")
    plt.ylabel("Profit")
    plt.title("Profit vs $\\delta_{eval}$")
    plt.legend()
    # Add both grid and minor grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(f"seed_{seed}/profit_vs_delta.png")
    plt.show()
