from mbdro.envs.simple import SimpleEnv

from mbdro.optimisation_mf import standard_mf
from mbdro.optimisation_mf import dro_xi_mf

from mbdro.optimisation_mb import standard_mb
from mbdro.optimisation_mb import dro_xi_mb
from mbdro.optimisation_mb import learn_conversion

import tqdm

import numpy as np
import torch


def single_error_evaluation(configuration):
    """Evaluate the error one time."""

    # Get alpha and beta from the configuration
    alpha = configuration["alpha"]
    beta = configuration["beta"]
    num_samples = configuration["num_samples"]
    action_range = configuration["action_range"]
    no_points = configuration["no_points"]
    delta = configuration.get("delta", 0.05)

    # Create the environment
    env = SimpleEnv(alpha=alpha, beta=beta)

    # Generate discrete data
    (prices, price_indices, outcomes, rewards) = env.generate_data_finite_A(
        num_samples, action_range=action_range, no_points=no_points
    )

    action_space = np.linspace(action_range[0], action_range[1], no_points)

    pi_0 = np.ones_like(action_space) / len(action_space)

    # Optimise using MF
    best_price_index, best_price, best_profit, profits = standard_mf.optimise_price_det(
        price_indices=price_indices,
        X=np.zeros_like(price_indices),
        y=outcomes,
        action_space=action_space,
        pi_0=pi_0,
    )

    best_price_index_dro, best_price_dro, best_profit_dro, profits_dro = (
        dro_xi_mf.optimise_robust_price_det(
            price_indices=price_indices,
            X=np.zeros_like(price_indices),
            y=outcomes,
            action_space=action_space,
            pi_0=pi_0,
            delta=delta,
        )
    )

    acceptance_model = learn_conversion.train_model(
        X=prices,
        y=outcomes,
        lr=0.05,
        no_iterations=4000,
        verbose=False,
        patience=400,
        tolerance=0.00001,
    )[0]

    # Compute the Linfty error at the action space
    true_probs_action_space = env.compute_acceptance_probability(action_space)
    true_profits_action_space = (
        env.compute_acceptance_probability(action_space) * action_space
    )
    true_dro_profits_action_space = true_profits_action_space - np.sqrt(
        delta
        * true_probs_action_space
        * (1 - true_probs_action_space)
        * action_space**2
    )

    action_tensor = torch.tensor(action_space, dtype=torch.float32)
    acceptance_prob_action_space = acceptance_model(action_tensor).detach().numpy()
    mb_profits_action_space = acceptance_prob_action_space * action_space
    mb_profits_dro_action_space = mb_profits_action_space - np.sqrt(
        delta
        * acceptance_prob_action_space
        * (1 - acceptance_prob_action_space)
        * action_space**2
    )

    # Compute the Linfty error for the DRO profits
    linf_error_mf = np.max(np.abs(profits_dro - true_dro_profits_action_space))
    linf_error_mb = np.max(
        np.abs(mb_profits_dro_action_space - true_dro_profits_action_space)
    )

    return (
        profits,
        profits_dro,
        mb_profits_action_space,
        mb_profits_dro_action_space,
        true_profits_action_space,
        true_dro_profits_action_space,
    )


def loop_error_evaluation(configuration, num_loops=10):
    """Evaluate the error multiple times."""
    aggr_profits = []
    aggr_profits_dro = []
    aggr_mb_profits_action_space = []
    aggr_mb_profits_dro_action_space = []
    aggr_true_profits_action_space = []
    aggr_true_dro_profits_action_space = []

    for _ in tqdm.tqdm(range(num_loops)):
        # Get alpha and beta from the configuration
        (
            profits,
            profits_dro,
            mb_profits_action_space,
            mb_profits_dro_action_space,
            true_profits_action_space,
            true_dro_profits_action_space,
        ) = single_error_evaluation(configuration)
        # Append the results
        aggr_profits.append(profits.tolist())
        aggr_profits_dro.append(profits_dro.tolist())
        aggr_mb_profits_action_space.append(mb_profits_action_space.tolist())
        aggr_mb_profits_dro_action_space.append(mb_profits_dro_action_space.tolist())
        aggr_true_profits_action_space.append(true_profits_action_space.tolist())
        aggr_true_dro_profits_action_space.append(
            true_dro_profits_action_space.tolist()
        )

    # Return all the results
    return (
        aggr_profits,
        aggr_profits_dro,
        aggr_mb_profits_action_space,
        aggr_mb_profits_dro_action_space,
        aggr_true_profits_action_space,
        aggr_true_dro_profits_action_space,
    )


def run_experiment(configuration, num_samples_arr, num_loops=10):
    results = {}
    for num_samples in num_samples_arr:
        configuration["num_samples"] = num_samples
        (
            aggr_profits,
            aggr_profits_dro,
            aggr_mb_profits_action_space,
            aggr_mb_profits_dro_action_space,
            aggr_true_profits_action_space,
            aggr_true_dro_profits_action_space,
        ) = loop_error_evaluation(configuration, num_loops=num_loops)
        results[num_samples] = {
            "profits": aggr_profits,
            "profits_dro": aggr_profits_dro,
            "mb_profits_action_space": aggr_mb_profits_action_space,
            "mb_profits_dro_action_space": aggr_mb_profits_dro_action_space,
            "true_profits_action_space": aggr_true_profits_action_space,
            "true_dro_profits_action_space": aggr_true_dro_profits_action_space,
        }

    return results


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Parameters for the experiment
    alpha = -15
    beta = 0.5
    action_range = (0.3, 0.7)
    no_points = 10
    delta = 0.05
    num_samples_arr = [70, 100, 200, 500, 1000, 2000, 5000]
    num_loops = 50

    configuration = {
        "alpha": alpha,
        "beta": beta,
        "action_range": action_range,
        "no_points": no_points,
        "delta": delta,
    }

    # Run the experiment
    results = run_experiment(configuration, num_samples_arr, num_loops=num_loops)

    # Print the results to a pickle file
    import pickle

    with open(f"results/results_{beta}.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to results.pkl")
