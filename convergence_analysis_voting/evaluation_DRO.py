# from policy import NeuralNetworkPolicy
import torch
import numpy as np

from learn_conversion import train_model


def evaluate_policy_model_dro(
    X, A_one_hot, labels, action_costs, actions_one_hot, delta
):
    """
    Evaluate the policy model by generating synthetic data and computing statistics.

    """
    X_augmented = np.hstack(
        [X, actions_one_hot]
    )  # shape: (n_samples, n_features + n_actions)
    # Train the logistic regression model
    model, mean_probs = train_model(X, A_one_hot, labels)
    # Apply the model to the X_augmented - policy application
    probs = model.predict_proba(X_augmented)[
        :, 1
    ]  # Get probabilities for the positive class
    # Compute the rewards
    costs = np.sum(action_costs * actions_one_hot, axis=1)
    rewards = probs - costs
    square_rewards = probs * (1-costs)**2 + (1-probs) * costs**2
    
    # Compute the mean reward
    mean_reward = np.mean(rewards)
    mean_square_reward = np.mean(square_rewards)

    # Compute the variance
    variance = mean_square_reward - mean_reward**2
    dro_reward = mean_reward - np.sqrt(delta * variance)
    # Return the mean reward
    return mean_reward, dro_reward


def evaluate_policy_data_dro(
    X, A_one_hot, labels, action_costs, actions_one_hot, delta
):
    """
    Evaluate the policy model by generating synthetic data and computing statistics.

    """
    rewards = labels - np.sum(action_costs * actions_one_hot, axis=1)
    # Find the indices where the action agrees with the  A_one_hot
    indicator = np.sum(A_one_hot * actions_one_hot, axis=1) > 0.5
    # Compute the rewards only for the actions that agree
    rewards = rewards[indicator == 1]
    # Compute the mean reward
    mean_reward = np.mean(rewards)
    variance = np.var(rewards)
    dro_reward = mean_reward - np.sqrt(delta * variance)
    # Return the mean reward
    return mean_reward, dro_reward


def apply_policy(X, policy):
    """
    Apply the policy to the generated data.

    Args:
        X : Input tensor of shape (batch_size, input_dim).
        policy (NeuralNetworkPolicy): Policy model.

    Returns:
        torch.Tensor: One-hot encoded actions.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_one_hot_policy = policy(X_tensor)
    # Choose the action with the highest probability
    actions = torch.argmax(A_one_hot_policy, dim=1)
    A_one_hot_policy = torch.nn.functional.one_hot(
        actions, num_classes=A_one_hot_policy.shape[1]
    )
    return A_one_hot_policy.numpy()
