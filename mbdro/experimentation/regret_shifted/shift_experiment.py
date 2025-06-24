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

import matplotlib.pyplot as plt


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def run_experiment(configuration):
    # Initialise samples of alpha and beta
    alpha = configuration["alpha"]
    beta = configuration["beta"]
    beta_variance = configuration["beta_variance"]
    num_samples = 5000
    price_range = np.linspace(0, 1, 1000)

    mle_rewards = price_range * sigmoid(alpha * (price_range - beta))


    # Sample the beta values
    beta_samples = np.random.normal(beta, beta_variance, num_samples)
    # sort the beta samples
    beta_samples = np.sort(beta_samples)

    # For each beta in the sample, find the optimal price using basic optimisation
    best_prices = []
    best_rewards = []
    for beta_sample in beta_samples:
        # Compute the reward for the given alpha and beta for each price in the price range
        rewards = price_range * sigmoid(alpha * (price_range - beta_sample))
        # Find the price that maximises the reward
        best_price = price_range[np.argmax(rewards)]
        best_reward = np.max(rewards)
        # Store the best price and reward
        best_prices.append(best_price)
        best_rewards.append(best_reward)

    plt.plot(beta_samples, best_prices, label="Best Price")
    plt.plot(beta_samples, best_rewards, label="Best Reward")
    plt.xlabel("Beta Sample")
    plt.ylabel("Value")
    plt.title(
        f"Best Price and Best Reward vs Beta Sample for alpha={alpha} and beta={beta}_variance={beta_variance}"
    )
    plt.legend()
    plt.grid()
    plt.show()

    # Convert to numpy arrays
    best_prices = np.array(best_prices)
    best_rewards = np.array(best_rewards)

    # Optimise for price which maximises the expected reward
    mean_rewards = []
    for price in price_range:
        mean_reward = np.mean(price * sigmoid(alpha * (price - beta_samples)))
        mean_rewards.append(mean_reward)

    # Convert to numpy array
    mean_rewards = np.array(mean_rewards)

    # Find the price that maximises the mean reward
    best_price = price_range[np.argmax(mean_rewards)]
    best_reward = np.max(mean_rewards)
    print(f"Best Price: {best_price}")
    print(f"Best Reward: {best_reward}")

    # Plot the reward distribution and the regret distribution
    rewards = best_price * sigmoid(alpha * (best_price - beta_samples))
    regret = best_rewards - rewards
    plt.hist(
        -regret,
        bins=50,
        density=True,
        alpha=0.5,
        label="Negative Regret Distribution",
    )
    plt.hist(
        rewards,
        bins=50,
        density=True,
        alpha=0.5,
        label="Reward Distribution",
    )
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"Objective Distribution for alpha={alpha} and beta={beta}_variance={beta_variance}")
    plt.legend()
    plt.grid()
    plt.savefig("reward_regret_distribution.png")
    plt.show()

    regret_price_range = np.linspace(best_price - 0.05, best_price + 0.05, 3)
    
    for price in regret_price_range:
        rewards = price * sigmoid(alpha * (price - beta_samples))
        regret = best_rewards - rewards
        plt.hist(
            -regret,
            bins=50,
            density=True,
            alpha=0.5,
            label=f"Negative regret Price: {price:.2f}",
        )
        plt.hist(
            rewards,
            bins=50,
            density=True,
            alpha=0.5,
            label=f"Reward Price: {price:.2f}",
        )
        
    plt.xlabel("Beta Sample")
    plt.ylabel("Negative Regret")
    plt.title(f"Objective distributions for alpha={alpha} and beta={beta}_variance={beta_variance}")
    plt.legend()
    plt.grid()
    plt.savefig("reward_regret_distributions.png")
    plt.show()

    mean_regret = []
    mean_rewards = []
    var_regret = []
    var_rewards = []
    for price in price_range:
        rewards = price * sigmoid(alpha * (price - beta_samples))
        regret = best_rewards - rewards
        mean_regret.append(np.mean(-regret))
        mean_rewards.append(np.mean(rewards))
        var_regret.append(np.var(-regret))
        var_rewards.append(np.var(rewards))

    # Convert to numpy arrays
    mean_regret = np.array(mean_regret)
    mean_rewards = np.array(mean_rewards)
    var_regret = np.array(var_regret)
    var_rewards = np.array(var_rewards)

    # # Plot the mean regret
    # plt.plot(price_range, mean_rewards, label="Mean Reward")
    # plt.plot(price_range, mle_rewards, label="MLE Reward")
    # plt.xlabel("Price")
    # plt.ylabel("Reward")
    # plt.title(f"Mean Reward vs MLE Reward for alpha={alpha} and beta={beta}_variance={beta_variance}")
    # plt.legend()
    # plt.grid()
    # plt.show()



    # Plot reget - var and reward - var
    plt.plot(price_range, mean_regret - np.sqrt(var_regret), label="Mean -Regret - Std")
    plt.plot(price_range, mean_regret, label="Mean -Regret")
    plt.plot(price_range, mean_rewards - np.sqrt(var_rewards), label="Mean Reward - Std")
    plt.plot(price_range, mean_rewards, label="Mean Reward")
    plt.xlabel("Price")
    plt.ylabel("Objective")
    plt.title(
        f"Robust objective alpha={alpha} and beta={beta}_variance={beta_variance}"
    )
    plt.legend()
    plt.grid()
    plt.savefig("robust_objective.png")
    plt.show()

    # Plot the robust regret for varying robustness parameter
    deltas = np.linspace(0, 1.5, 5)
    for delta in deltas:
        robust_regret = mean_regret -  np.sqrt(delta *var_regret)
        plt.plot(price_range, robust_regret, label=f"Robust Regret (delta={delta})")
    plt.xlabel("Price")
    plt.ylabel("Robust Regret")
    plt.xlim(0.1, 0.4)
    plt.title(
        f"Robust Regret for alpha={alpha} and beta={beta}_variance={beta_variance}"
    )
    plt.legend()
    plt.grid()
    plt.savefig("robust_regret.png")
    plt.show()





if __name__ == "__main__":
    seed = 0
    # Parameters for the experiment
    alpha = -15
    beta = 0.3

    beta_variance = 0.1

    configuration = {
        "alpha": alpha,
        "beta": beta,
        "beta_variance": beta_variance,
    }

    # Run the experiment
    run_experiment(configuration)
