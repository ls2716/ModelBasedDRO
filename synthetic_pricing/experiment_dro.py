"""Implementation of a shift experiment to compare different DRO methods.

In this experiment, we will compare the performance of different DRO pricing
optimisation methods in a situation where the bias of the demand is shifted.

A single experiment will be as follows:
Input:
- alpha: the alpha value for the DRO method
- beta: the beta value for the DRO method
- betas: the betas value for the DRO method
- num_samples: the number of samples to use for the experiment
- price range: the range of prices to use for the experiment data
- num_sims: the number of simulations to run for the alpha and beta
Algorithm:
1. Generate a sparse dataset with the specified parameters.
2. For each method, run the pricing optimisation algorithm with the generated data.
3. Evaluate the method on the true parameters.
4. Store the results for each method.
5. Return the results for each method.
Output:
- results: a dictionary containing the results for each method, including the
  method name, the alpha and beta values used, and the performance metrics.
"""

import numpy as np
import torch
import pickle
import os

import mbdro.optimisation_mb.dro_KL_mb as dro_KL_mb
import mbdro.optimisation_mb.standard_mb as standard_mb
import mbdro.optimisation_mb.learn_conversion as mb_conversion

import matplotlib.pyplot as plt

# Increase the default font size for matplotlib
plt.rcParams.update({"font.size": 14})


def run_single(
    alpha,
    beta,
    delta_mb_arr,
    verbose=False,
    seed=None,
    initial_price=0.2,
):
    """Run a single experiment with the given parameters.

    Args:
        alpha (float): The alpha value for the DRO method.
        beta (float): The beta value for the DRO method.
        evaluation_model (str): The model to use for evaluation.
        delta_mb_arr (list): The delta values for the MB method.
        verbose (bool): Whether to print verbose output.
        seed (int): The random seed for reproducibility.
    Returns:
        result: (dict): A dictionary with results profit evaluations
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize the acceptance model for the MB method
    mb_acceptance_model = mb_conversion.LogisticModel()
    mb_acceptance_model.alpha.data = torch.tensor(alpha, dtype=torch.float32)
    mb_acceptance_model.beta.data = torch.tensor(beta, dtype=torch.float32)

    optimised_price_mb = standard_mb.optimise_price(
        acceptance_model=mb_acceptance_model,
        initial_price=initial_price,
        verbose=verbose,
    )

    optimised_robust_prices_mb = []
    for delta_mb in delta_mb_arr:
        robust_price_mb = dro_KL_mb.optimise_robust_price(
            acceptance_model=mb_acceptance_model,
            initial_price=optimised_price_mb,
            delta=delta_mb,
            learning_rate=0.01,
            num_iterations=200,
            verbose=verbose,
        )
        optimised_robust_prices_mb.append(robust_price_mb)
    print("Optimised robust prices for MB method for all deltas")
    print("------------------------")
    deltas = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.12, 0.15])
    robust_profits_mb = {}
    for delta_mb, optimised_robust_price_mb in zip(
        delta_mb_arr, optimised_robust_prices_mb
    ):
        r_profits = []

        for delta in deltas:
            if delta == 0:
                # For delta = 0, we use the standard evaluation
                r_profit = standard_mb.evaluate_profit(
                    optimised_robust_price_mb,
                    mb_acceptance_model,
                )
            else:
                r_profit = dro_KL_mb.evaluate_robust_profit(
                    optimised_robust_price_mb,
                    mb_acceptance_model,
                    delta=delta,
                )
            r_profits.append(r_profit)
        robust_profits_mb[delta_mb] = np.array(r_profits).flatten()

    robust_profits_standard = []
    for delta in deltas:
        if delta == 0:
            # For delta = 0, we use the standard price
            r_profit = standard_mb.evaluate_profit(
                optimised_price_mb,
                mb_acceptance_model,
            )
        else:
            r_profit = dro_KL_mb.evaluate_robust_profit(
                optimised_price_mb,
                mb_acceptance_model,
                delta=delta,
            )
        robust_profits_standard.append(r_profit)
    robust_profits_standard = np.array(robust_profits_standard).flatten()

    # Plot the results
    plot_path = f"plots/experiment_dro_alpha_{alpha}_beta_{beta}.png"

    plt.figure(figsize=(10, 6))
    plt.title("$V^{DRO}$ vs $\\delta_{eval}$")
    plt.xlabel("$\\delta_{eval}$")
    plt.ylabel("$V^{DRO}$")
    plt.plot(
        deltas,
        robust_profits_standard,
        "o-",
        label="standard (non-robust)",
    )
    for delta_mb, robust_profits in robust_profits_mb.items():
        plt.plot(
            deltas,
            robust_profits,
            "o-",
            # label="$V^{DRO,\\delta}(p^{DRO,\\varepsilon})$ for" + f" $\\varepsilon$ = {delta_mb:.3f}",
            label="$\\delta_{train}=$" + f"{delta_mb:.3f}",
        )
    plt.legend()
    # Add grid with minor ticks
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(plot_path)
    plt.show()
    plt.close()


if __name__ == "__main__":
    seed = 0
    # Parameters for the experiment
    alpha = -15
    beta = 0.0

    delta_mb_arr = [0.002, 0.005, 0.02, 0.05, 0.1]

    configuration = {
        "alpha": alpha,
        "beta": beta,
        "delta_mb_arr": delta_mb_arr,
        "seed": seed,
    }

    # Run the experiment
    run_single(
        configuration["alpha"],
        configuration["beta"],
        delta_mb_arr,
        verbose=True,
        seed=seed,
    )
