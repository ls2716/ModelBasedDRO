import pytest
from mbdro.envs.simple import SimpleEnv
import numpy as np


# Test the SimpleEnv class
def test_simple_env_init():
    env = SimpleEnv(alpha=-10, beta=0.5)
    assert env.alpha == -10
    assert env.beta == 0.5
    assert isinstance(env, SimpleEnv)


@pytest.fixture(scope="module")
def simple_env():
    """Fixture to create a SimpleEnv instance."""
    return SimpleEnv(alpha=-10, beta=0.5)


def test_compute_acceptance_probability(simple_env):
    price = 0.5
    acceptance_prob = simple_env.compute_acceptance_probability(price)
    assert 0 <= acceptance_prob <= 1, (
        "Acceptance probability should be between 0 and 1."
    )
    assert isinstance(acceptance_prob, float), (
        "Acceptance probability should be a float."
    )


def test_compute_acceptance_probability_array(simple_env):
    prices = np.array([0.5, 0.7, 0.9]).reshape(-1, 1)
    acceptance_probs = simple_env.compute_acceptance_probability(prices)
    assert acceptance_probs.shape == prices.shape, (
        "Output shape should match input shape."
    )
    assert np.all(acceptance_probs >= 0) and np.all(acceptance_probs <= 1), (
        "Acceptance probabilities should be between 0 and 1."
    )


def test_generate_data(simple_env):
    num_samples = 10
    prices, outcomes, rewards = simple_env.generate_data(num_samples=num_samples)
    assert prices.shape == (num_samples, 1), "Prices shape should be (num_samples, 1)."
    assert outcomes.shape == (num_samples, 1), (
        "Outcomes shape should be (num_samples, 1)."
    )
    assert rewards.shape == (num_samples, 1), (
        "Rewards shape should be (num_samples, 1)."
    )
    assert np.allclose(rewards, outcomes * prices), (
        "Rewards should be equal to outcomes multiplied by prices."
    )
    assert np.all(np.isin(outcomes, [0, 1])), "Outcomes should be binary (0 or 1)."


def test_reprodubility():
    # Set the random seed for reproducibility
    np.random.seed(0)
    env = SimpleEnv(alpha=-10, beta=0.5)
    prices, outcomes, rewards = env.generate_data(num_samples=5)
    # Assert that first prices is around 0.5488135
    assert np.isclose(prices[0, 0], 0.5488135, atol=1e-5), (
        "First price should be around 0.5488135"
    )
    assert outcomes[0, 0] == 1, "First outcome should be 1"


def test_generate_data_finite_A(simple_env):
    num_samples = 10
    action_range = (0, 1)
    no_points = 5
    prices, price_indices, outcomes, rewards = simple_env.generate_data_finite_A(
        num_samples=num_samples, action_range=action_range, no_points=no_points
    )
    action_space = np.linspace(
        action_range[0], action_range[1], no_points, endpoint=True
    )
    assert prices.shape == (num_samples, 1), "Prices shape should be (num_samples, 1)."
    assert outcomes.shape == (num_samples, 1), (
        "Outcomes shape should be (num_samples, 1)."
    )
    assert rewards.shape == (num_samples, 1), (
        "Rewards shape should be (num_samples, 1)."
    )
    assert price_indices.shape == (num_samples, 1), (    
            "Price indices shape should be (num_samples, 1)."
        )
    assert np.allclose(rewards, outcomes * prices), (
        "Rewards should be equal to outcomes multiplied by prices."
    )
    assert np.all(np.isin(outcomes, [0, 1])), "Outcomes should be binary (0 or 1)."
    assert np.all(prices >= action_range[0]) and np.all(prices <= action_range[1]), (
        "Prices should be within the action range."
    )
    assert np.all(np.isin(prices, action_space)), (
        "Prices should be sampled from the action space."
    )
