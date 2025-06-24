"""In this script, for the given environment and the data, we will plot the estimated
profits and robust profits for a range of prices.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from mbdro.envs.simple import SimpleEnv
import mbdro.optimisation_mf.standard_mf as opt_standard_mf
import mbdro.optimisation_mf.dro_xi_mf as opt_dro_xi_mf
import mbdro.optimisation_mb.dro_xi_mb as opt_dro_xi_mb
import mbdro.optimisation_mb.standard_mb as opt_standard_mb
import mbdro.optimisation_mb.learn_conversion as opt_conversion_mb
import mbdro.optimisation_bayesian.learn_conversion as opt_conversion_bayesian
import mbdro.optimisation_bayesian.dro_xi_bayesian as opt_dro_xi_bayesian
import mbdro.optimisation_bayesian.standard_expectation as opt_standard_bayesian
import mbdro.optimisation_bayesian.standard_mle as opt_standard_mle_bayesian


if __name__ == "__main__":
    NUM_SAMPLES = 1000
    NUM_POSTERIOR_SAMPLES = 1000
    NUM_PRICES = 11
    PRICE_RANGE = (0.1, 0.4)
    DELTA_MF = 0.1
    DELTA_MB = 0.005
    DELTA_BAYESIAN = 1.

    SEED = 2

    ALPHA = -10
    BETA = 0.3

    # Set the seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = SimpleEnv(alpha=ALPHA, beta=BETA)
    prices_X, price_indices, outcomes_y, reward_outcomes = env.generate_data_finite_A(
        num_samples=NUM_SAMPLES, action_range=PRICE_RANGE, no_points=NUM_PRICES
    )

    action_space = np.linspace(
        PRICE_RANGE[0], PRICE_RANGE[1], NUM_PRICES, endpoint=True
    ).reshape(-1, 1)
    prices_plot = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
    true_acceptance_probabilities = env.compute_acceptance_probability(prices_plot)
    true_rewards = prices_plot * true_acceptance_probabilities

    # Evaluate the profits using model-free approach standard and robust
    pi_0 = np.ones_like(action_space) / action_space.shape[0]
    profits_mf = np.zeros_like(action_space)
    robust_profits_mf = np.zeros_like(action_space)
    for index in range(NUM_PRICES):
        profits_mf[index] = opt_standard_mf.evaluate_profit_det(
            index, price_indices, prices_X, outcomes_y, action_space, pi_0
        )
        robust_profits_mf[index] = opt_dro_xi_mf.evaluate_robust_profit_det(
            index, price_indices, prices_X, outcomes_y, action_space, pi_0, DELTA_MF
        )
    
    # Now use the model-based approach
    # Train the acceptance model using the training data
    mb_acceptance_model, training_info = opt_conversion_mb.train_model(prices_X, outcomes_y, patience=100)
    # Evaluate profits
    profits_mb = opt_standard_mb.evaluate_profit(prices_plot, mb_acceptance_model)
    robust_profits_mb = opt_dro_xi_mb.evaluate_robust_profit(
        prices_plot, mb_acceptance_model, DELTA_MB
    )

    # Now the Bayesian approach
    # Train the acceptance model using the training data
    bayesian_acceptance_model, posterior_samples = opt_conversion_bayesian.train_model(
        prices_X, outcomes_y, num_samples=NUM_POSTERIOR_SAMPLES
    )
    # Evaluate profits
    profits_bayesian = opt_standard_bayesian.evaluate_profit(
        prices_plot, bayesian_acceptance_model
    )
    robust_profits_bayesian = opt_dro_xi_bayesian.evaluate_robust_profit(
        prices_plot, bayesian_acceptance_model, DELTA_BAYESIAN
    )
    mle_profits_bayesian = opt_standard_mle_bayesian.evaluate_mle_profit(
        prices_plot, bayesian_acceptance_model
    )

    # Plot the true acceptance probabilities and rewards
    plt.figure(figsize=(12, 6))
    plt.plot(prices_plot, true_rewards, label="True Rewards")
    plt.scatter(
        action_space, profits_mf, label="model-free profit"
    )
    plt.scatter(
        action_space, robust_profits_mf, label="model-free robust profit"
    )
    plt.plot(prices_plot, profits_mb, label="model-based profit")
    plt.plot(prices_plot, robust_profits_mb, label="model-based robust profit")
    plt.plot(
        prices_plot, profits_bayesian, label="bayesian profit"
    )
    plt.plot(
        prices_plot, robust_profits_bayesian, label="bayesian robust profit"
    )
    plt.plot(
        prices_plot, mle_profits_bayesian, label="bayesian mle profit"
    )
    plt.xlabel("Price")
    plt.ylabel("Reward")
    plt.title("True Rewards vs Price")
    plt.legend()
    plt.grid()
    plt.show()


    # Plot the acceptance probabilities of the true model, standard model and bayesian model
    prices_plot_pt = torch.tensor(prices_plot)
    mb_acceptance_probabilities = mb_acceptance_model(prices_plot_pt).detach().numpy()
    bayesian_acceptance_probabilities = bayesian_acceptance_model(prices_plot_pt).detach().numpy()
    # Sample columns for plotting the bayesian posterior
    no_samples = 40
    sampled_indices = np.random.choice(
        NUM_POSTERIOR_SAMPLES, no_samples, replace=False
    )
    
    plt.figure(figsize=(12, 6))
    
    
    for i in sampled_indices:
        plt.plot(
            prices_plot,
            bayesian_acceptance_probabilities[:,i],
            # label=f"Bayesian Acceptance Probabilities {i}",
            alpha=0.2,
            color="green",
        )
    plt.plot(prices_plot, mb_acceptance_probabilities, label="Model-based Acceptance Probabilities")
    plt.plot(prices_plot, true_acceptance_probabilities, label="True Acceptance Probabilities")

    plt.xlabel("Price")
    plt.ylabel("Acceptance Probability")
    plt.title("Acceptance Probabilities vs Price")
    plt.legend()
    plt.grid()
    plt.show()

