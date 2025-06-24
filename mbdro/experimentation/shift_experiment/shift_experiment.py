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
import pickle
import os

import mbdro.optimisation_bayesian.dro_xi_bayesian as dro_xi_bayesian
import mbdro.optimisation_bayesian.standard_expectation as standard_expectation
import mbdro.optimisation_bayesian.learn_conversion as bayesian_conversion
import mbdro.optimisation_mb.dro_xi_mb as dro_xi_mb
import mbdro.optimisation_mb.standard_mb as standard_mb
import mbdro.optimisation_mb.learn_conversion as mb_conversion

from plot_results import plot_results


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


def loop(alphas, betas, beta_variance, delta_mb_arr, delta_bayesian_arr, seed):
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
    
    evaluation_model = bayesian_conversion.generate_model(
        mean_alpha=alpha,
        mean_beta=beta,
        variance_alpha=0.0,
        variance_beta=beta_variance,
        correlation=0.0,
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
    return results
    

def run_experiment(configuration):
    """Run the experiment with the given configuration.

    Args:
        configuration (dict): The configuration for the experiment.
    Returns:
        results: (dict): A dictionary with results profit evaluations
    """
    alpha = configuration["alpha"]
    beta = configuration["beta"]
    beta_variance = configuration["beta_variance"]

    delta_mb_arr = configuration["delta_mb_arr"]
    delta_bayesian_arr = configuration["delta_bayesian_arr"]
    seed = configuration["seed"]

    results_dictionary = loop(
        alphas=alpha,
        betas=beta,
        beta_variance=beta_variance,
        delta_mb_arr=delta_mb_arr,
        delta_bayesian_arr=delta_bayesian_arr,
        seed=seed,
    )

    metadata = {
        "alpha": alpha,
        "beta": beta,
        "beta_variance": beta_variance,
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
        results_folder, f"shift_experiment_results_{seed}_{beta}_{beta_variance}.pkl"
    )
    with open(results_file, "wb") as f:
        pickle.dump(output, f)
    
    # # Try unpickling the results
    # with open(results_file, "rb") as f:
    #     loaded_output = pickle.load(f)
    # print("Loaded output:", loaded_output)

    # plot_results(output["results"], plot_path=None)
    return output
    





if __name__ == "__main__":
    seed = 0
    # Parameters for the experiment
    alpha = -15
    beta = 0.3

    beta_variance = 0.05
    delta_mb_arr = [0.02, 0.05, 0.1, 0.2, 0.5, 1.]
    delta_bayesian_arr = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    configuration = {
        "alpha": alpha,
        "beta": beta,
        "beta_variance": beta_variance,
        "delta_mb_arr": delta_mb_arr,
        "delta_bayesian_arr": delta_bayesian_arr,
        "seed": seed,
    }

    # Run the experiment
    run_experiment(configuration)


