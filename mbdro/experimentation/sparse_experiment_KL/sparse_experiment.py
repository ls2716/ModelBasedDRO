"""Implementation of a sparse experiment to compare different DRO methods.

In this experiment, we will compare the performance of different DRO pricing
optimisation methods in a situation where data is sparse.

A single experiment will be as follows:
Input:
- alpha: the alpha value for the DRO method
- beta: the beta value for the DRO method
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
import matplotlib.pyplot as plt

from mbdro.envs.simple import SimpleEnv
import mbdro.optimisation_bayesian.dro_xi_bayesian as dro_xi_bayesian
import mbdro.optimisation_bayesian.standard_expectation as standard_expectation
import mbdro.optimisation_bayesian.learn_conversion as bayesian_conversion
import mbdro.optimisation_mb.dro_KL_mb as dro_KL_mb
import mbdro.optimisation_mb.standard_mb as standard_mb
import mbdro.optimisation_mb.learn_conversion as mb_conversion


def run_single(
    alpha,
    beta,
    num_samples,
    price_range,
    delta_mb,
    delta_bayesian,
    verbose=False,
    seed=None,
):
    """Run a single experiment with the given parameters.

    Args:
        alpha (float): The alpha value for the DRO method.
        beta (float): The beta value for the DRO method.
        num_samples (int): The number of samples to use for the experiment.
        price_range (tuple): The range of prices to use for the experiment data.
    Returns:
        result: (dict): A dictionary with results (true profit)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Create the environment
    env = SimpleEnv(
        alpha=alpha,
        beta=beta,
    )
    # Generate data
    X, y, r = env.generate_data(
        num_samples=num_samples,
        price_range=price_range,
    )
    initial_price = 0.2

    # Run the model_based optimisation
    mb_acceptance_model, training_info = mb_conversion.train_model(
        X=X, y=y, verbose=verbose
    )
    optimised_price_mb = standard_mb.optimise_price(
        acceptance_model=mb_acceptance_model,
        initial_price=initial_price,
        verbose=verbose,
    )
    acceptance_prob_mb = env.compute_acceptance_probability(price=optimised_price_mb)
    profit_mb = float(acceptance_prob_mb * optimised_price_mb)
    print("Profit MB: ", profit_mb)
    optimised_robust_price_mb = dro_KL_mb.optimise_robust_price(
        acceptance_model=mb_acceptance_model,
        initial_price=initial_price,
        delta=delta_mb,
        verbose=verbose,
    )
    acceptance_prob_robust_mb = env.compute_acceptance_probability(
        price=optimised_robust_price_mb
    )
    profit_robust_mb = float(acceptance_prob_robust_mb * optimised_robust_price_mb)
    print("Profit Robust MB: ", profit_robust_mb)

    bayesian_acceptance_model, samples = bayesian_conversion.train_model(
        X=X,
        y=y,
        num_samples=1000,
    )
    optimised_price_bayesian = standard_expectation.bayesian_optimise_price(
        acceptance_model=bayesian_acceptance_model,
        initial_price=initial_price,
        verbose=verbose,
    )
    acceptance_prob_bayesian = env.compute_acceptance_probability(
        price=optimised_price_bayesian
    )
    profit_bayesian = float(acceptance_prob_bayesian * optimised_price_bayesian)
    print("Profit Bayesian: ", profit_bayesian)
    optimised_robust_price_bayesian = dro_xi_bayesian.bayesian_optimise_robust_price(
        acceptance_model=bayesian_acceptance_model,
        initial_price=initial_price,
        delta=delta_bayesian,
        verbose=verbose,
    )
    acceptance_prob_robust_bayesian = env.compute_acceptance_probability(
        price=optimised_robust_price_bayesian
    )
    profit_robust_bayesian = float(
        acceptance_prob_robust_bayesian * optimised_robust_price_bayesian
    )
    print("Profit Robust Bayesian: ", profit_robust_bayesian)

    return {
        "price_mb": optimised_price_mb,
        "profit_mb": profit_mb,
        "price_robust_mb": optimised_robust_price_mb,
        "profit_robust_mb": profit_robust_mb,
        "price_bayesian": optimised_price_bayesian,
        "profit_bayesian": profit_bayesian,
        "price_robust_bayesian": optimised_robust_price_bayesian,
        "profit_robust_bayesian": profit_robust_bayesian,
    }


def loop(
    alpha,
    beta,
    num_samples,
    price_range,
    delta_mb,
    delta_bayesian,
    num_sims=10,
    verbose=False,
    base_seed=None,
):
    """Run the experiment for multiple simulations."""
    results = []
    for i in range(num_sims):
        result = run_single(
            alpha=alpha,
            beta=beta,
            num_samples=num_samples,
            price_range=price_range,
            delta_mb=delta_mb,
            delta_bayesian=delta_bayesian,
            verbose=verbose,
            seed=base_seed + i * 100,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    seed = 10
    no_sims = 20
    # Parameters for the experiment
    alpha = -10
    beta = 0.2
    num_samples = 150
    price_range = (0.2, 0.4)
    delta_mb = 0.1
    delta_bayesian = 8.0

    # seed = 10
    # alpha = -20
    # beta = 0.2
    # num_samples = 100
    # price_range = (0.1, 0.9)
    # delta_mb = 0.4
    # delta_bayesian = 6.

    # Run the experiment
    results = run_single(
        alpha=alpha,
        beta=beta,
        num_samples=num_samples,
        price_range=price_range,
        delta_mb=delta_mb,
        delta_bayesian=delta_bayesian,
        verbose=False,
        seed=seed,
    )

    # print("Results: ", results)
    results = loop(
        alpha=alpha,
        beta=beta,
        num_samples=num_samples,
        price_range=price_range,
        delta_mb=delta_mb,
        delta_bayesian=delta_bayesian,
        num_sims=no_sims,
        verbose=False,
        base_seed=seed,
    )

    # Collect the standard profits
    standard_mb_profits = [result["profit_mb"] for result in results]
    robust_mb_profits = [result["profit_robust_mb"] for result in results]
    standard_bayesian_profits = [result["profit_bayesian"] for result in results]
    robust_bayesian_profits = [result["profit_robust_bayesian"] for result in results]

    # Print the mean, std and the 5% quantile of the profits
    print(
        "Standard MB Profit: ",
        np.mean(standard_mb_profits),
        np.std(standard_mb_profits),
        np.quantile(standard_mb_profits, 0.05),
    )
    print(
        "Robust MB Profit: ",
        np.mean(robust_mb_profits),
        np.std(robust_mb_profits),
        np.quantile(robust_mb_profits, 0.05),
    )
    print(
        "Standard Bayesian Profit: ",
        np.mean(standard_bayesian_profits),
        np.std(standard_bayesian_profits),
        np.quantile(standard_bayesian_profits, 0.05),
    )
    print(
        "Robust Bayesian Profit: ",
        np.mean(robust_bayesian_profits),
        np.std(robust_bayesian_profits),
        np.quantile(robust_bayesian_profits, 0.05),
    )

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(standard_mb_profits, label="Standard Profit MB")
    # plt.plot(robust_mb_profits, label="Robust Profit MB")
    plt.plot(standard_bayesian_profits, label="Standard Profit Bayesian")
    plt.plot(robust_bayesian_profits, label="Robust Profit Bayesian")
    plt.xlabel("Simulation")
    plt.ylabel("Profit")
    plt.title("Profit Comparison between Standard and Robust MB")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot histogram of the profits
    plt.figure(figsize=(10, 5))
    plt.hist(standard_mb_profits, bins=10, alpha=0.5, label="Standard Profit MB")
    plt.hist(robust_mb_profits, bins=10, alpha=0.5, label="Robust Profit MB")
    plt.hist(
        standard_bayesian_profits, bins=10, alpha=0.5, label="Standard Profit Bayesian"
    )
    plt.hist(
        robust_bayesian_profits, bins=10, alpha=0.5, label="Robust Profit Bayesian"
    )
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.title("Profit Distribution")
    plt.legend()
    plt.grid()
    plt.show()

    # Save the results as json with metadata containing the parameters used
    import json
    import os
    import datetime

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(
        results_dir,
        f"results_{alpha}_{beta}_{num_samples}_{price_range[0]}_{price_range[1]}_{delta_mb}_{delta_bayesian}_{seed}_{no_sims}.json",
    )

    metadata = {
        "alpha": alpha,
        "beta": beta,
        "num_samples": num_samples,
        "price_range": price_range,
        "delta_mb": delta_mb,
        "delta_bayesian": delta_bayesian,
        "seed": seed,
        "no_sims": no_sims,
        "date": str(datetime.datetime.now()),
    }
    with open(results_file, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=4)
    print(f"Results saved to {results_file}")
    print("Experiment completed.")
