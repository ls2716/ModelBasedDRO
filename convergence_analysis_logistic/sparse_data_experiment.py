from mbdro.envs.simple import SimpleEnv

from mbdro.optimisation_mf import standard_mf
from mbdro.optimisation_mf import dro_xi_mf

from mbdro.optimisation_mb import standard_mb
from mbdro.optimisation_mb import dro_xi_mb
from mbdro.optimisation_mb import learn_conversion

import numpy as np
import torch


def run_experiment(configuration):
    # Get alpha and beta from the configuration
    alpha = configuration["alpha"]
    beta = configuration["beta"]
    num_samples = configuration["num_samples"]
    action_range = configuration["action_range"]
    no_points = configuration["no_points"]
    delta = configuration.get("delta", 0.1)

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
            delta=0.1,
        )
    )

    acceptance_model = learn_conversion.train_model(
        X=prices,
        y=outcomes,
        lr=0.1,
        no_iterations=1000,
        verbose=False,
        patience=100,
        tolerance=0.000001,
    )[0]

    # Plot the profits vs the prices
    price_range = np.linspace(0, 1, 100)
    true_probs = env.compute_acceptance_probability(price_range)
    true_profits = env.compute_acceptance_probability(price_range) * price_range
    true_dro_profits = true_profits - np.sqrt(0.1 * true_probs * (1 - true_probs)*price_range**2)

    # Compute the profits using the fitted model
    price_tensor = torch.tensor(price_range, dtype=torch.float32)
    acceptance_prob = acceptance_model(price_tensor).detach().numpy()
    mb_profits = acceptance_prob * price_range
    mb_profits_dro = mb_profits - np.sqrt(0.1 * acceptance_prob * (1 - acceptance_prob) * price_range**2)

    import matplotlib.pyplot as plt

    # Increase the size of the font
    plt.rcParams.update({"font.size": 16})

    plt.figure(figsize=(8, 5))
    plt.scatter(action_space, profits, label="Data-based standard")
    plt.plot(price_range, mb_profits, label="Parametric standard")
    plt.scatter(action_space, profits_dro, label="Data-based DRO")
    plt.plot(price_range, mb_profits_dro, label="Parametric DRO")
    plt.plot(price_range, true_profits, label="True standard")
    plt.plot(price_range, true_dro_profits, label="True DRO ")
    plt.xlabel("Price")
    plt.ylabel("Objective")
    plt.title("Objective evaluation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("mf_vs_mb_example.png")
    plt.show()

    # Compute the Linfty error at the action space
    true_probs_action_space = env.compute_acceptance_probability(action_space)
    true_profits_action_space = env.compute_acceptance_probability(action_space) * action_space
    true_dro_profits_action_space = true_profits_action_space - np.sqrt(
        0.1 * true_probs_action_space * (1 - true_probs_action_space) * action_space**2
    )

    action_tensor = torch.tensor(action_space, dtype=torch.float32)
    acceptance_prob_action_space = acceptance_model(action_tensor).detach().numpy()
    mb_profits_action_space = acceptance_prob_action_space * action_space
    mb_profits_dro_action_space = mb_profits_action_space - np.sqrt(
        0.1 * acceptance_prob_action_space * (1 - acceptance_prob_action_space) * action_space**2
    )

    # Compute the Linfty error for the DRO profits
    linf_error_mf = np.max(np.abs(profits_dro - true_dro_profits_action_space))
    linf_error_mb = np.max(np.abs(mb_profits_dro_action_space - true_dro_profits_action_space))
    print(f"Linfty error for MF: {linf_error_mf}")
    print(f"Linfty error for MB: {linf_error_mb}")



if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Parameters for the experiment
    alpha = -10
    beta = 0.3
    num_samples = 500
    action_range = (0.1, 0.5)
    no_points = 10
    delta = 0.05

    configuration = {
        "alpha": alpha,
        "beta": beta,
        "num_samples": num_samples,
        "action_range": action_range,
        "no_points": no_points,
        "delta": delta,
    }

    # Run the experiment
    run_experiment(configuration)
