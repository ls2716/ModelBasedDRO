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

from mbdro.envs.simple import SimpleEnv
import mbdro.optimisation_bayesian.dro_xi_bayesian as dro_xi_bayesian
import mbdro.optimisation_bayesian.standard_expectation as standard_expectation
import mbdro.optimisation_bayesian.learn_conversion as bayesian_conversion
import mbdro.optimisation_mb.dro_KL_mb as dro_KL_mb
import mbdro.optimisation_mb.standard_mb as standard_mb
import mbdro.optimisation_mb.learn_conversion as mb_conversion

from plot_result import plot_results

def run_single(
    alpha,
    beta,
    num_samples,
    price_range,
    delta_mb_arr,
    delta_bayesian_arr,
    verbose=False,
    seed=None,
    initial_price=0.2,
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

    optimisted_robust_price_mb_dict = {}
    for delta_mb in delta_mb_arr:
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
        print(f"Profit Robust MB for delta={delta_mb}: ", profit_robust_mb)

        optimisted_robust_price_mb_dict[delta_mb] = (
            optimised_robust_price_mb,
            profit_robust_mb,
        )

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

    optimised_robust_price_bayesian_dict = {}

    for delta_bayesian in delta_bayesian_arr:
        optimised_robust_price_bayesian = (
            dro_xi_bayesian.bayesian_optimise_robust_price(
                acceptance_model=bayesian_acceptance_model,
                initial_price=initial_price,
                delta=delta_bayesian,
                verbose=verbose,
            )
        )
        acceptance_prob_robust_bayesian = env.compute_acceptance_probability(
            price=optimised_robust_price_bayesian
        )
        profit_robust_bayesian = float(
            acceptance_prob_robust_bayesian * optimised_robust_price_bayesian
        )
        print(
            f"Profit Robust Bayesian for delta={delta_bayesian}: ",
            profit_robust_bayesian,
        )

        optimised_robust_price_bayesian_dict[delta_bayesian] = (
            optimised_robust_price_bayesian,
            profit_robust_bayesian,
        )

    return {
        "seed": seed,
        "mb": (optimised_price_mb, profit_mb),
        "robust_mb": optimisted_robust_price_mb_dict,
        "bayesian": (optimised_price_bayesian, profit_bayesian),
        "robust_bayesian": optimised_robust_price_bayesian_dict,
    }


def loop(
    alpha,
    beta,
    num_samples,
    price_range,
    delta_mb_arr,
    delta_bayesian_arr,
    num_sims=5,
    verbose=False,
    base_seed=None,
    initial_price=0.2,
):
    """Run the experiment for multiple simulations."""
    results = []
    for i in range(num_sims):
        result = run_single(
            alpha=alpha,
            beta=beta,
            num_samples=num_samples,
            price_range=price_range,
            delta_mb_arr=delta_mb_arr,
            delta_bayesian_arr=delta_bayesian_arr,
            verbose=verbose,
            seed=base_seed + i * 100,
            initial_price=initial_price,
        )
        results.append(result)

    return results


def compute_optimal_price(alpha, beta, plot=False):
    """Compute the optimal price for the given alpha and beta."""
    # Compute the optimal price using line search
    price_range = np.linspace(0.0, 2.0, num=201)
    acceptance_prob = 1 / (1 + np.exp(-alpha * (price_range - beta)))
    profit = price_range * acceptance_prob
    optimal_price = price_range[np.argmax(profit)]
    optimal_profit = profit.max()
    return optimal_price, optimal_profit


def run_experiment(configuration):
    """Run the experiment with the given configuration."""
    alpha = configuration["alpha"]
    beta = configuration["beta"]
    num_samples = configuration["num_samples"]
    num_sims = configuration["num_sims"]
    verbose = configuration["verbose"]
    seed = configuration["seed"]
    price_range_multiplier = configuration["price_range_multiplier"]

    # Compute the optimal price for the given alpha and beta
    optimal_price, optimal_profit = compute_optimal_price(alpha, beta)

    # Compute the price range as (1-multiplier)*optimal_price to (1+multiplier)*optimal_price
    price_range = (
        (1 - price_range_multiplier) * optimal_price,
        (1 + price_range_multiplier) * optimal_price,
    )

    print("Price range: ", price_range)
    print("Optimal price: ", optimal_price)

    # HARD CODED BAYESIAN DELTA
    delta_bayesian_arr = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0]
    delta_mb_arr = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    metadata = {
        "alpha": alpha,
        "beta": beta,
        "num_samples": num_samples,
        "price_range_multiplier": price_range_multiplier,
        "optimal_price": optimal_price,
        "price_range": price_range,
        "delta_bayesian_arr": delta_bayesian_arr,
        "delta_mb_arr": delta_mb_arr,
        "num_sims": num_sims,
        "seed": seed,
    }

    results = loop(
        alpha=alpha,
        beta=beta,
        num_samples=num_samples,
        price_range=price_range,
        delta_mb_arr=delta_mb_arr,
        delta_bayesian_arr=delta_bayesian_arr,
        num_sims=num_sims,
        verbose=verbose,
        base_seed=seed,
        initial_price=optimal_price,
    )

    output = {
        "metadata": metadata,
        "results": results,
    }

    # Save the results to a json file
    import json
    import os

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(
        results_dir,
        f"results_{alpha}_{beta}_{num_samples}_{price_range_multiplier}.json",
    )
    with open(results_file, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {results_file}")

    plot_results(results)

    return results



if __name__ == "__main__":
    ## TESTING
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    alpha = -15.0
    beta = 0.3
    num_samples = 150

    no_sims = 100
    results = run_experiment(
        configuration={
            "alpha": alpha,
            "beta": beta,
            "num_samples": num_samples,
            "num_sims": no_sims,
            "verbose": False,
            "seed": seed,
            "price_range_multiplier": 0.5,
        }
    )
    print("Results: ", results)

    # Open configuration file from command line arguments
    # import argparse
    # import json

    # parser = argparse.ArgumentParser(description="Run the sparse experiment.")
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     required=True,
    #     help="Path to the configuration file.",
    # )
    # args = parser.parse_args()

    # with open(args.config, "r") as f:
    #     configuration = json.load(f)
    # print("Configuration: ", configuration)

    # run_experiment(configuration)
