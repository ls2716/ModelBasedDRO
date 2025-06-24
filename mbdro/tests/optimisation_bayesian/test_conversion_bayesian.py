"""Test bayesian conversion module."""

import pytest

import numpy as np
import torch

from mbdro.optimisation_bayesian.learn_conversion import train_model, extend_posterior
from mbdro.envs.simple import SimpleEnv


@pytest.fixture(scope="module")
def data():
    """Fixture to generate data."""
    # set the seed
    seed = 0
    np.random.seed(seed)
    env = SimpleEnv(alpha=-10, beta=0.5)
    X, y, r = env.generate_data(num_samples=100)
    return X, y


def test_train_model(data):
    """Test the training of the model."""

    X, y = data
    # Train the acceptance model using the training data
    acceptance_model, posterior_samples = train_model(X, y, num_samples=500)
    assert acceptance_model is not None, "Acceptance model should not be None."
    assert posterior_samples is not None, "Posterior samples should not be None."
    assert len(posterior_samples) > 0, "Posterior samples should not be empty."
    assert posterior_samples["alpha"].shape[0] == 500, (
        "Posterior samples should have the correct shape."
    )
    assert posterior_samples["beta"].shape[0] == 500, (
        "Posterior samples should have the correct shape."
    )

    # Compute mean and variance of the posterior samples
    mean_alpha = np.mean(posterior_samples["alpha"])
    mean_beta = np.mean(posterior_samples["beta"])
    var_alpha = np.var(posterior_samples["alpha"])
    var_beta = np.var(posterior_samples["beta"])

    print(f"Alpha samples mean: {mean_alpha:.3f}, var: {var_alpha:.5f}")
    print(f"Beta samples mean: {mean_beta:.3f}, var: {var_beta:.5f}")

    # Check that mean alpha and beta are close to the true values
    assert np.isclose(mean_alpha, -10, rtol=0.1), (
        "Mean alpha is not close to true value."
    )
    assert np.isclose(mean_beta, 0.5, rtol=0.1), "Mean beta is not close to true value."

    # Assert that vairances are larger than zero 0.0001
    assert var_alpha > 0.0001, "Variance of alpha samples is too small."
    assert var_beta > 0.0001, "Variance of beta samples is too small."


@pytest.fixture(scope="module")
def model_samples(data):
    """Fixture to generate samples."""
    X, y = data
    # Train the acceptance model using the training data
    acceptance_model, posterior_samples = train_model(X, y, num_samples=500)
    return acceptance_model, posterior_samples


def test_extension(model_samples):
    """Test sample variance extension."""

    # First compute the mean and covariance matrix for the samples
    acceptance_model, posterior_samples = model_samples
    all_samples = np.concatenate(
        [
            posterior_samples["alpha"].reshape(-1, 1),
            posterior_samples["beta"].reshape(-1, 1),
        ],
        axis=1,
    )
    # Compute the means
    means = np.mean(all_samples, axis=0)
    # Compute the covariance matrix
    covariance_matrix = np.cov(all_samples, rowvar=False)
    print("Original means")
    print(means)
    print("Original covariance matrix")
    print(covariance_matrix)

    # First extend by a factor of 1.0
    extended_samples = extend_posterior(
        posterior_samples, no_new_samples=500, alpha_factor=1.0, beta_factor=1.0
    )
    # Compute the new means and covariance matrix
    extended_all_samples = np.concatenate(
        [
            extended_samples["alpha"].reshape(-1, 1),
            extended_samples["beta"].reshape(-1, 1),
        ],
        axis=1,
    )
    extended_means = np.mean(extended_all_samples, axis=0)
    extended_covariance_matrix = np.cov(extended_all_samples, rowvar=False)
    print("Extended means")
    print(extended_means)
    print("Extended covariance matrix")
    print(extended_covariance_matrix)

    # Assert almost equal means
    assert np.allclose(means, extended_means, rtol=0.1), (
        "Means are not equal."
    )
    # Assert almost equal covariance matrices
    assert np.allclose(covariance_matrix, extended_covariance_matrix, rtol=0.2), (
        "Covariance matrices are not equal."
    )


def test_extension_f2(model_samples):
    """"Test sample variance extension by factors."""
    # First compute the mean and covariance matrix for the samples
    alpha_factor = 2.0
    beta_factor = 1.5
    acceptance_model, posterior_samples = model_samples
    all_samples = np.concatenate(
        [
            posterior_samples["alpha"].reshape(-1, 1),
            posterior_samples["beta"].reshape(-1, 1),
        ],
        axis=1,
    )
    # Compute the means
    means = np.mean(all_samples, axis=0)
    # Compute the covariance matrix
    covariance_matrix = np.cov(all_samples, rowvar=False)
    print("Original means")
    print(means)
    print("Original covariance matrix")
    print(covariance_matrix)

    # First extend by a factors
    extended_samples = extend_posterior(
        posterior_samples, no_new_samples=1000, alpha_factor=alpha_factor, beta_factor=beta_factor
    )
    # Compute the new means and covariance matrix
    extended_all_samples = np.concatenate(
        [
            extended_samples["alpha"].reshape(-1, 1),
            extended_samples["beta"].reshape(-1, 1),
        ],
        axis=1,
    )
    extended_means = np.mean(extended_all_samples, axis=0)
    extended_covariance_matrix = np.cov(extended_all_samples, rowvar=False)
    print("Extended means")
    print(extended_means)
    print("Extended covariance matrix")
    print(extended_covariance_matrix)

    # Check the correct values
    # Assert almost equal means
    assert np.allclose(means, extended_means, rtol=0.1), (
        "Means are not equal."
    )
    # Assert correct relation between the covariance matrices
    assert np.isclose(covariance_matrix[0, 0] * alpha_factor, extended_covariance_matrix[0, 0], rtol=0.1), (
        "Covariance matrix for alpha is not correct."
    )
    assert np.isclose(covariance_matrix[1, 1] * beta_factor, extended_covariance_matrix[1, 1], rtol=0.1), (
        "Covariance matrix for beta is not correct."
    )
    assert np.isclose(
        covariance_matrix[0, 1] * np.sqrt(alpha_factor * beta_factor),
        extended_covariance_matrix[0, 1],
        rtol=0.1,
    ), "Covariance matrix for alpha and beta is not correct."
    assert np.isclose(
        covariance_matrix[1, 0] * np.sqrt(alpha_factor * beta_factor),
        extended_covariance_matrix[1, 0],
        rtol=0.1,
    ), "Covariance matrix for beta and alpha is not correct."


def test_mle(model_samples):
    """Test the MLE of the model."""
    acceptance_model, posterior_samples = model_samples
    # Compute the MLE parameters
    alpha_mle = acceptance_model.alpha_mle
    beta_mle = acceptance_model.beta_mle

    # Check that the MLE parameters are close to the true values
    assert np.isclose(alpha_mle, -10, rtol=0.1), (
        "MLE alpha is not close to true value."
    )
    assert np.isclose(beta_mle, 0.5, rtol=0.1), "MLE beta is not close to true value."


def test_acceptance(model_samples):
    acceptance_model, posterior_samples = model_samples
    num_samples = posterior_samples['alpha'].shape[0]
    # Test the acceptance probability function and a array of prices
    prices = np.linspace(0, 1, 11, endpoint=True)
    prices_pt = torch.tensor(prices, dtype=torch.float32).reshape(-1, 1)
    acceptance_prob = acceptance_model(prices_pt)
    assert acceptance_prob.shape == (11, num_samples), "Acceptance probability shape is incorrect."