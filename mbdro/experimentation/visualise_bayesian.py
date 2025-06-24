"""Code to visualise the reward distribution according to the Bayesian model."""

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from mbdro.envs.simple import SimpleEnv
import mbdro.optimisation_bayesian.dro_xi_bayesian as dro_xi_bayesian
import mbdro.optimisation_bayesian.standard_expectation as standard_expectation
import mbdro.optimisation_bayesian.learn_conversion as bayesian_conversion


def plot_reward_dist(
    acceptance_model, delta_bayesian, verbose=False, seed=None, initial_price=0.2
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
    price_standard = standard_expectation.bayesian_optimise_price(
        acceptance_model=acceptance_model,
        initial_price=initial_price,
        verbose=verbose,
    )
    price_robust = dro_xi_bayesian.bayesian_optimise_robust_price(
        acceptance_model=acceptance_model,
        initial_price=initial_price,
        delta=delta_bayesian,
        verbose=verbose,
    )

    # Compute reward distribution for the prices
    acceptance_prob_standard = acceptance_model(
        torch.tensor(price_standard).reshape(1, 1)
    ).flatten()
    acceptance_prob_robust = acceptance_model(
        torch.tensor(price_robust).reshape(1, 1)
    ).flatten()

    reward_standard = price_standard * acceptance_prob_standard
    reward_robust = price_robust * acceptance_prob_robust

    # Plot the reward distribution
    plt.figure(figsize=(10, 6))
    # Compute the bins for the histplot
    min_reward = min(reward_robust.min(), reward_standard.min())
    max_reward = max(reward_standard.max(), reward_robust.max())
    no_bins = 40
    bins = list(np.linspace(min_reward, max_reward, num=no_bins+1, endpoint=True))
    sns.histplot(reward_standard, bins=bins, kde=True, stat='density', label="Standard price")
    sns.histplot(reward_robust, bins=bins, kde=True, stat='density', label="Robust price")
    plt.xlabel("Price")
    plt.ylabel("Reward")
    plt.title("Reward Distribution")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Set the seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ALPHA = -3.0
    BETA = 0.3
    NUM_DATA = 80
    PRICE_RANGE = (0.4, 0.6)
    NUM_SAMPLES = 5000
    DELTA_BAYESIAN = 4.

    # Create images folder
    images_folder = "images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Initialise the environment
    env = SimpleEnv(alpha=ALPHA, beta=BETA)
    # Generate the data
    X, y, r = env.generate_data(num_samples=NUM_DATA)

    # Plot acceptance probability
    price_range = np.linspace(0,1,100)
    probs = env.compute_acceptance_probability(price_range).flatten()
    rewards = probs * price_range
    plt.plot(price_range, probs, label="Acceptance")
    plt.plot(price_range, rewards, label="Reward")
    plt.xlabel("Price")
    plt.ylabel("Acceptance_Probability/ Reward")
    plt.legend()
    plt.grid()
    plt.show()

    # Train the model
    acceptance_model, posterior_samples = bayesian_conversion.train_model(
        X, y, num_samples=NUM_SAMPLES
    )
    bayesian_conversion.print_info(posterior_samples)

    plot_reward_dist(acceptance_model=acceptance_model, delta_bayesian=DELTA_BAYESIAN)


    # # Generate artificial model with uncertainty in beta only
    # acceptance_model, samples = bayesian_conversion.generate_model(
    #     mean_alpha=ALPHA,
    #     mean_beta=BETA,
    #     variance_alpha=1.,
    #     variance_beta=0.05,
    #     correlation=0.0,
    #     num_samples=NUM_SAMPLES
    # )

    # plot_reward_dist(acceptance_model=acceptance_model, delta_bayesian=DELTA_BAYESIAN)
