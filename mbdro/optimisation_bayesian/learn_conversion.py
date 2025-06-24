"""Implement baysian model of conversion probability."""

import numpy as np
import torch

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS


def get_samples(X, y, num_samples=2000, num_warmup=500):
    # Convert to JAX arrays
    X = jnp.array(X)
    y = jnp.array(y.flatten())

    # Define Bayesian Logistic Model
    def model(X, y=None):
        alpha = numpyro.sample(
            "alpha",
            dist.Uniform(jnp.zeros(X.shape[1]) - 40.0, jnp.zeros(X.shape[1])-1),
        )
        beta = numpyro.sample("beta", dist.Normal(0.5, 10.0))
        logits = jnp.dot(X - beta, alpha.T)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    # Run MCMC
    rng_key = random.PRNGKey(0)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X, y)

    return mcmc.get_samples()


class AcceptanceModel:
    def __init__(self, samples):
        self.alpha = torch.tensor(samples["alpha"], dtype=torch.float32)
        self.beta = torch.tensor(samples["beta"], dtype=torch.float32).reshape(-1, 1)
        # Compute maximum likelihood estimates (MLE) for alpha and beta
        self.alpha_mle = self.alpha.mean()
        self.beta_mle = self.beta.mean()

    def __call__(self, X):
        # Compute the logits
        logits = (X - self.beta.T) * self.alpha.T
        # Compute the probabilities using sigmoid function
        probs = torch.sigmoid(logits)
        return probs


def train_model(X, y, num_samples, num_warmup=500):
    # Train the model and get samples
    samples = get_samples(X, y, num_samples, num_warmup)
    # Create the acceptance model
    model = AcceptanceModel(samples)
    return model, samples


def print_info(samples):
    """Print the summary of the samples."""
    # Print the summary of the samples
    print("Summary of the samples:")
    # Print mean and var for each weight
    for key in samples.keys():
        mean = samples[key].mean()
        var = samples[key].var()
        print(f"{key}: mean={mean:.4f}, var={var:.4f}")
    # Print covariance matrix between weights and bias
    all_samples = np.concatenate(
        [samples["alpha"].reshape(-1, 1), samples["beta"].reshape(-1, 1)], axis=1
    )
    covariance_matrix = np.cov(all_samples, rowvar=False)
    print("Covariance matrix:")
    print(covariance_matrix)
    # Print the correlation between alpha and beta
    correlation = covariance_matrix[0, 1] / (
        np.sqrt(covariance_matrix[0, 0]) * np.sqrt(covariance_matrix[1, 1])
    )
    print(f"Correlation between alpha and beta: {correlation:.2f}")


def extend_posterior(samples, no_new_samples=None, alpha_factor=1.0, beta_factor=1.0):
    """Extend the posterior samples by multiplying the variance by factors."""
    # Compute the mean for the alpha and beta
    alpha_mean = samples["alpha"].mean()
    beta_mean = samples["beta"].mean()
    means = np.array([alpha_mean, beta_mean])
    # Compute the covariance matrix for the alpha and beta
    all_samples = np.concatenate(
        [samples["alpha"].reshape(-1, 1), samples["beta"].reshape(-1, 1)], axis=1
    )
    # Step 1: Compute covariance matrix and std devs
    covariance_matrix = np.cov(all_samples, rowvar=False)
    std_devs = np.sqrt(np.diag(covariance_matrix))

    # Step 2: Compute correlation matrix
    corr_matrix = covariance_matrix / np.outer(std_devs, std_devs)

    # Step 3: Scale std devs
    new_std_devs = std_devs.copy()
    new_std_devs[0] *= np.sqrt(alpha_factor)
    new_std_devs[1] *= np.sqrt(beta_factor)

    # Step 4: Reconstruct new covariance matrix
    new_covariance_matrix = corr_matrix * np.outer(new_std_devs, new_std_devs)

    # Step 5: Generate new samples
    if no_new_samples is None:
        no_new_samples = samples["alpha"].shape[0]

    new_samples = np.random.multivariate_normal(
        means, new_covariance_matrix, size=no_new_samples
    )

    return {
        "alpha": new_samples[:, 0].reshape(-1, 1),
        "beta": new_samples[:, 1].flatten(),
    }


def generate_model(
    mean_alpha, mean_beta, variance_alpha, variance_beta, correlation, num_samples
):
    """Generate a model with the given parameters."""

    # Compute the covariance matrix
    covariance_matrix = np.array(
        [
            [variance_alpha, correlation * np.sqrt(variance_alpha * variance_beta)],
            [correlation * np.sqrt(variance_alpha * variance_beta), variance_beta],
        ]
    )
    means = np.array([mean_alpha, mean_beta])
    # Generate samples from the multivariate normal distribution
    samples = np.random.multivariate_normal(means, covariance_matrix, size=num_samples)
    samples_dict = {
        "alpha": samples[:, 0].reshape(-1, 1),
        "beta": samples[:, 1].reshape(-1),
    }
    # Define the acceptance model
    acceptance_model = AcceptanceModel(samples_dict)
    # Return the model and the samples
    return acceptance_model, samples_dict

def generate_model_uniform(
    alpha_l, alpha_r, beta_l, beta_r, num_samples
):
    """Generate a model with the given parameters."""

    # Generate samples from the uniform distribution
    samples = np.random.uniform(
        low=np.array([alpha_l, beta_l]), high=np.array([alpha_r, beta_r]), size=(num_samples, 2)
    )
    # Reshape the samples to match the expected format
    samples_dict = {
        "alpha": samples[:, 0].reshape(-1, 1),
        "beta": samples[:, 1].reshape(-1),
    }
    # Define the acceptance model
    acceptance_model = AcceptanceModel(samples_dict)
    # Return the model and the samples
    return acceptance_model, samples_dict


if __name__ == "__main__":
    # Set the seed
    seed = 0
    import numpy as np

    np.random.seed(seed)

    # Create images folder
    import os

    images_folder = "images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Import the simple environment
    from mbdro.envs.simple import SimpleEnv

    # Initialise the environment
    env = SimpleEnv(alpha=-20, beta=0.5)
    # Generate the data
    X, y = env.generate_data(num_samples=200)

    # Train the model
    acceptance_model, posterior_samples = train_model(
        X, y, num_samples=10000, num_warmup=500
    )
    print(posterior_samples["alpha"].shape)

    # Plot the acceptance probability as a function of price
    price_range = np.linspace(0, 1, 100).reshape(-1, 1)
    # Compute probs
    probs = acceptance_model(torch.tensor(price_range, dtype=torch.float32)).numpy()

    # Plot the acceptance curves for the price range
    # Sample 40 parameters from the posterior
    num_samples = 200
    indices = np.random.choice(
        posterior_samples["alpha"].shape[0], num_samples, replace=False
    )
    probs_sampled = probs[:, indices]
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.plot(price_range.flatten(), probs_sampled[:, i].flatten(), alpha=0.5)

    plt.grid()
    plt.title("Acceptance probability as a function of price")
    plt.xlabel("Price")
    plt.ylabel("Acceptance probability")
    plt.savefig("images/acceptance_probability_data.png")
    plt.close()

    # Print summary of samples (means and variances)
    print("Posterior samples:")
    print_info(posterior_samples)

    # Generate new posterior samples
    new_samples = extend_posterior(posterior_samples, alpha_factor=2, beta_factor=2)
    print(new_samples["alpha"].shape)
    # Print the information about the new samples
    print("Extended samples:")
    print_info(new_samples)

    # Plot the distribution of the alphas and beta on 2d histogram using two subplots
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the distribution of original posterior samples on the left
    h = sns.histplot(
        x=posterior_samples["alpha"].flatten(),
        y=posterior_samples["beta"].flatten(),
        bins=30,
        ax=ax[0],
        cmap="Blues",
        cbar=True,
        stat="density",
        fill=True,
    )
    # Get vmax from the histogram
    quadmesh = h.collections[0]
    vmax = quadmesh.get_clim()[1]
    ax[0].set_title("Original Posterior Samples")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("beta")
    ax[0].set_xlim(np.min(new_samples["alpha"]), np.max(new_samples["alpha"]))
    ax[0].set_ylim(np.min(new_samples["beta"]), np.max(new_samples["beta"]))
    ax[0].grid()
    # Plot the distribution of new posterior samples on the right
    sns.histplot(
        x=new_samples["alpha"].flatten(),
        y=new_samples["beta"].flatten(),
        bins=30,
        ax=ax[1],
        cmap="Blues",
        cbar=True,
        stat="density",
        vmax=vmax,
        fill=True,
    )
    ax[1].set_title("Extended Posterior Samples")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("beta")
    ax[1].set_xlim(np.min(new_samples["alpha"]), np.max(new_samples["alpha"]))
    ax[1].set_ylim(np.min(new_samples["beta"]), np.max(new_samples["beta"]))
    ax[1].grid()
    plt.tight_layout()
    plt.savefig("images/posterior_samples.png")
    plt.close()

    new_model, new_samples = generate_model(
        mean_alpha=-20,
        mean_beta=0.5,
        variance_alpha=4.0,
        variance_beta=0.005,
        correlation=-0.1,
        num_samples=10000,
    )
    print("New model samples:")
    print_info(new_samples)
    # Plot the acceptance probability as a function of price
    price_range = np.linspace(0, 1, 100).reshape(-1, 1)
    # Compute probs
    probs = new_model(torch.tensor(price_range, dtype=torch.float32)).numpy()
    # Sample 40 parameters from the posterior
    num_samples = 40
    indices = np.random.choice(
        new_samples["alpha"].shape[0], num_samples, replace=False
    )
    probs_sampled = probs[:, indices]
    # Plot the acceptance curves for the price range
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.plot(price_range.flatten(), probs_sampled[:, i].flatten(), alpha=0.5)
    plt.grid()
    plt.title("Acceptance probability as a function of price for artificial model")
    plt.xlabel("Price")
    plt.ylabel("Acceptance probability")
    plt.savefig("images/acceptance_probability_artificial.png")
    plt.close()
