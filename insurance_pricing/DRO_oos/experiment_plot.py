"""Plot the results of the experiment."""


import os
import matplotlib.pyplot as plt
import json

# Increase the default font size for better readability
plt.rcParams.update({'font.size': 16})



if __name__=="__main__":
    
    ratio = 0.9
    
    # seeds = [0, 1, 2, 3, 4,5,6,7,8,9]
    # seeds =  [10,11,12,13,14,15,16,17,18,19]
    seeds = [20,21,22,23,24,25,26,27,28,29]
    # seeds = [30,31,32,33,34,35,36,37,38,39]
    # seeds = [40,41,42,43,44,45,46,47,48,49]
    seeds = list(range(30))  # Use all 50 seeds for robustness

    deltas = [0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.05]#, 0.1]
    prices_to_evaluate = [
        "standard",
        "robust_delta_0.001",
        "robust_delta_0.002",
        "robust_delta_0.003",
        "robust_delta_0.005",
        "robust_delta_0.01",
        "robust_delta_0.05",
        # "robust_delta_0.1",
    ]

    # Initialise the plot
    plt.figure(figsize=(10, 6))
    for seed in seeds:
        # Load the evaluation dictionary
        evaluation_dict_path = f"ratio_{ratio}/seed_{seed}/evaluation_dict.json"
        if not os.path.exists(evaluation_dict_path):
            print(f"Evaluation dictionary for seed {seed} not found. Skipping...")
            continue
        with open(evaluation_dict_path, "r") as f:
            evaluation_dict = json.load(f)
        # Extract the profits for each delta
        profits = []
        for i in range(len(deltas)):
            price_key = prices_to_evaluate[i]
            if price_key in evaluation_dict:
                profits.append(evaluation_dict[price_key])
            else:
                profits.append(None)

        # Plot the profits vs deltas for each seed
        plt.plot(deltas, profits, marker='o', label=f'Seed {seed}')
    plt.xlabel('Ambiguity radius $\\delta$')
    plt.ylabel('Mean profit')
    plt.title('Mean Profit vs $\\delta$ for 9:1 Train/Test Split')
    # Add grid with minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.xticks(deltas)
    plt.minorticks_on()
    plt.legend(loc='upper right')
    plt.tight_layout()
    # Save the plot
    plt.savefig(f'ratio_{ratio}/ratio_{ratio}_mean_profit_vs_delta.png')
    plt.show()
    print(f"Plot saved as 'ratio_{ratio}/ratio_{ratio}_mean_profit_vs_delta.png'.")
