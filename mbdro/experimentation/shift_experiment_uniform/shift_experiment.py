"""Implementation of a shift experiment to compare different DRO methods.

In this experiment, we will compare the performance of different DRO pricing
optimisation methods in a situation where the bias of the demand is shifted.

A single experiment will be as follows:
Input:
- alpha: the alpha value for the DRO method
- beta: the beta value for the DRO method
- betas: the betas value for the DRO method
- bayesian_posterior: the posterior distribution for the Bayesian method
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
import pickle
import os

import mbdro.optimisation_bayesian.dro_xi_bayesian as dro_xi_bayesian
import mbdro.optimisation_bayesian.standard_expectation as standard_expectation
import mbdro.optimisation_bayesian.learn_conversion as bayesian_conversion
import mbdro.optimisation_mb.dro_xi_mb as dro_xi_mb
import mbdro.optimisation_mb.standard_mb as standard_mb
import mbdro.optimisation_mb.learn_conversion as mb_conversion


def run_single(
    alpha,
    beta,
    evaluation_model,
    bayesian_model,
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
        evaluation_model (str): The model to use for evaluation.
        bayesian_model (str): The model to use for the Bayesian method.
        delta_mb_arr (list): The delta values for the MB method.
        delta_bayesian_arr (list): The delta values for the Bayesian method.
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
    # Evaluate using the evaluation model
    acceptance_distr_mb = (
        evaluation_model(torch.tensor(optimised_price_mb).reshape(1, 1))
        .flatten()
        .numpy()
    )
    profit_distr_mb = optimised_price_mb * acceptance_distr_mb

    profit_distr_robust_mb_dict = {}
    for delta_mb in delta_mb_arr:
        optimised_robust_price_mb = dro_xi_mb.optimise_robust_price(
            acceptance_model=mb_acceptance_model,
            initial_price=optimised_price_mb,
            delta=delta_mb,
            verbose=verbose,
        )
        # Evaluate using the evaluation model
        acceptance_distr_robust_mb = (
            evaluation_model(torch.tensor(optimised_robust_price_mb).reshape(1, 1))
            .flatten()
            .numpy()
        )
        profit_distr_robust_mb = acceptance_distr_robust_mb * optimised_robust_price_mb
        profit_distr_robust_mb_dict[delta_mb] = (
            optimised_robust_price_mb,
            profit_distr_robust_mb,
        )

    # Run the Bayesian method
    optimised_price_bayesian = standard_expectation.bayesian_optimise_price(
        acceptance_model=bayesian_model,
        initial_price=initial_price,
    )
    # Evaluate using the evaluation model
    acceptance_distr_bayesian = (
        evaluation_model(torch.tensor(optimised_price_bayesian).reshape(1, 1))
        .flatten()
        .numpy()
    )
    profit_distr_bayesian = optimised_price_bayesian * acceptance_distr_bayesian
    profit_distr_robust_bayesian_dict = {}
    for delta_bayesian in delta_bayesian_arr:
        optimised_robust_price_bayesian = (
            dro_xi_bayesian.bayesian_optimise_robust_price(
                acceptance_model=bayesian_model,
                initial_price=optimised_price_bayesian,
                delta=delta_bayesian,
            )
        )
        # Evaluate using the evaluation model
        acceptance_distr_robust_bayesian = (
            evaluation_model(
                torch.tensor(optimised_robust_price_bayesian).reshape(1, 1)
            )
            .flatten()
            .numpy()
        )
        profit_distr_robust_bayesian = (
            acceptance_distr_robust_bayesian * optimised_robust_price_bayesian
        )
        profit_distr_robust_bayesian_dict[delta_bayesian] = (
            optimised_robust_price_bayesian,
            profit_distr_robust_bayesian,
        )

    return {
        "result_mb": (optimised_price_mb, profit_distr_mb),
        "result_robust_mb": profit_distr_robust_mb_dict,
        "result_bayesian": (optimised_price_bayesian, profit_distr_bayesian),
        "result_robust_bayesian": profit_distr_robust_bayesian_dict,
    }


def loop(alphas, betas, beta_width, delta_mb_arr, delta_bayesian_arr, seed):
    """Run the experiment with the given parameters.

    Args:
        alphas (list): The alpha values for the DRO method.
        betas (list): The beta values for the DRO method.
        beta_variance (float): The variance of the beta value.
        delta_mb_arr (list): The delta values for the MB method.
        delta_bayesian_arr (list): The delta values for the Bayesian method.
        seed (int): The random seed for reproducibility.
    Returns:
        results: (dict): A dictionary with results profit evaluations
    """
    results_dict = {}
    for alpha in alphas:
        for beta in betas:
            # Generate the evaluation model
            alpha_l = alpha
            alpha_r = alpha
            beta_l = beta - beta_width / 2
            beta_r = beta + beta_width / 2
            evaluation_model = bayesian_conversion.generate_model_uniform(
                alpha_l=alpha_l,
                alpha_r=alpha_r,
                beta_l=beta_l,
                beta_r=beta_r,
                num_samples=40000,
            )[0]

            results = run_single(
                alpha=alpha,
                beta=beta,
                evaluation_model=evaluation_model,
                bayesian_model=evaluation_model,
                delta_mb_arr=delta_mb_arr,
                delta_bayesian_arr=delta_bayesian_arr,
                verbose=False,
                seed=seed,
            )
            results_dict[str(alpha) + "_" + str(beta)] = results
    return results_dict


def run_experiment(configuration):
    """Run the experiment with the given configuration.

    Args:
        configuration (dict): The configuration for the experiment.
    Returns:
        results: (dict): A dictionary with results profit evaluations
    """
    alphas = configuration["alphas"]
    betas = configuration["betas"]
    beta_width = configuration["beta_width"]

    delta_mb_arr = configuration["delta_mb_arr"]
    delta_bayesian_arr = configuration["delta_bayesian_arr"]
    seed = configuration["seed"]

    results_dictionary = loop(
        alphas=alphas,
        betas=betas,
        beta_width=beta_width,
        delta_mb_arr=delta_mb_arr,
        delta_bayesian_arr=delta_bayesian_arr,
        seed=seed,
    )

    metadata = {
        "alphas": alphas,
        "betas": betas,
        "beta_width": beta_width,
        "delta_mb_arr": delta_mb_arr,
        "delta_bayesian_arr": delta_bayesian_arr,
        "seed": seed,
    }

    # Save the results pickle file
    output = {
        "metadata": metadata,
        "results": results_dictionary,
    }
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_file = os.path.join(
        results_folder, f"shift_experiment_results_{seed}_{beta_width}.pkl"
    )
    with open(results_file, "wb") as f:
        pickle.dump(output, f)

    # # Try unpickling the results
    # with open(results_file, "rb") as f:
    #     loaded_output = pickle.load(f)
    # print("Loaded output:", loaded_output)

    plot_results(output["results"])
    return output


def plot_results(results):
    """Plot the results of the experiment.

    Args:
        results (dict): The results of the experiment.
    """
    # Plot the results here
    for key, value in results.items():
        # Get alpha and beta from the key
        alpha, beta = map(float, key.split("_"))
        result_mb = value["result_mb"]
        result_bayesian = value["result_bayesian"]
        result_robust_mb = value["result_robust_mb"]
        result_robust_bayesian = value["result_robust_bayesian"]

        # Plot the mean and std for the profit distributions
        mean_profit_mb = np.mean(result_mb[1])
        std_profit_mb = np.std(result_mb[1])
        mean_profit_bayesian = np.mean(result_bayesian[1])
        std_profit_bayesian = np.std(result_bayesian[1])

        plt.figure(figsize=(10, 6))
        plt.title(f"Profit Mean vs Std for alpha={alpha}, beta={beta}")
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
                xytext=(5, 5),
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
                xytext=(5, 5),
                ha="left",
                fontsize=8,
                color="tab:orange",
            )

        plt.legend()
        plt.grid()
        plt.show()

        # Plot the mean and percentile for the profit distributions
        percentile = 10
        mean_profit_mb = np.mean(result_mb[1])
        perc5_profit_mb = np.percentile(result_mb[1], percentile)

        mean_profit_bayesian = np.mean(result_bayesian[1])
        perc5_profit_bayesian = np.percentile(result_bayesian[1], percentile)

        plt.figure(figsize=(10, 6))
        plt.title(
            f"Profit Mean vs {percentile}% Percentile for alpha={alpha}, beta={beta}"
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
                xytext=(5, 5),
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
                xytext=(5, 5),
                ha="left",
                fontsize=8,
                color="tab:orange",
            )

        # Set aspect ratio to 1:1
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot the profit distributions for MB, Bayesian and robust methods with delta 0.2
        delta_to_plot = 0.05
        # Choose bins based on the min and max profit values for MB and Bayesian method
        min_profit = min(result_mb[1].min(), result_bayesian[1].min())
        max_profit = max(result_mb[1].max(), result_bayesian[1].max())
        no_bins = 80
        bins = list(np.linspace(min_profit, max_profit, num=no_bins + 1, endpoint=True))
        plt.figure(figsize=(10, 6))
        plt.title(
            f"Profit Distribution for alpha={alpha}, beta={beta}, delta={delta_to_plot}"
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
        plt.show()


if __name__ == "__main__":
    seed = 0
    # Parameters for the experiment
    alphas = [-15.0]
    beta = [0.1]

    beta_width = 0.05
    delta_mb_arr = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    delta_bayesian_arr = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    configuration = {
        "alphas": alphas,
        "betas": beta,
        "beta_width": beta_width,
        "delta_mb_arr": delta_mb_arr,
        "delta_bayesian_arr": delta_bayesian_arr,
        "seed": seed,
    }

    # Run the experiment
    run_experiment(configuration)
