import numpy as np
import torch
from generate_data import generate_data
from policy import NeuralNetworkPolicy


def true_evaluation(seed, policy, action_costs, no_samples, generation_seed=0):
    """
    Evaluate the data generation process by generating synthetic data and computing statistics.

    Args:
        seed (int): Random seed for reproducibility.
        policy (str): Policy type for data generation.
        no_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the generated data and computed statistics.
    """
    # Generate synthetic data - only gather contexts
    X, _, _, gen_dict = generate_data(seed, no_samples, generation_seed=generation_seed)

    # Apply the policy to the generated data
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_one_hot_policy = policy(X_tensor)
    # Choose the action with the highest probability
    actions = torch.argmax(A_one_hot_policy, dim=1)
    A_one_hot_policy = torch.nn.functional.one_hot(
        actions, num_classes=A_one_hot_policy.shape[1]
    )

    # Compute the true rewards
    weights = torch.tensor(gen_dict["weights"], dtype=torch.float32)
    bias = torch.tensor(gen_dict["bias"], dtype=torch.float32)
    # Create an augmented input for the logistic function
    X_augmented = torch.cat(
        (X_tensor, A_one_hot_policy), dim=1
    )  # shape: (n_samples, n_features + n_actions)
    logits = torch.matmul(X_augmented, weights) + bias
    probs = torch.sigmoid(logits)

    # Sample labels from Bernoulli using the computed probabilities
    rewards = probs - torch.sum(torch.tensor(action_costs) * A_one_hot_policy, dim=1)

    mean_reward = torch.mean(rewards).item()

    # Return the mean reward
    return mean_reward

def true_evaluation_dro(seed, policy, action_costs, no_samples, delta, generation_seed=0):
    """
    Evaluate the data generation process by generating synthetic data and computing statistics.

    Args:
        seed (int): Random seed for reproducibility.
        policy (str): Policy type for data generation.
        no_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the generated data and computed statistics.
    """
    # Generate synthetic data - only gather contexts
    X, _, _, gen_dict = generate_data(seed, no_samples, generation_seed=generation_seed)

    # Apply the policy to the generated data
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_one_hot_policy = policy(X_tensor)
    # Choose the action with the highest probability
    actions = torch.argmax(A_one_hot_policy, dim=1)
    A_one_hot_policy = torch.nn.functional.one_hot(
        actions, num_classes=A_one_hot_policy.shape[1]
    )

    # Compute the true rewards
    weights = torch.tensor(gen_dict["weights"], dtype=torch.float32)
    bias = torch.tensor(gen_dict["bias"], dtype=torch.float32)
    # Create an augmented input for the logistic function
    X_augmented = torch.cat(
        (X_tensor, A_one_hot_policy), dim=1
    )  # shape: (n_samples, n_features + n_actions)
    logits = torch.matmul(X_augmented, weights) + bias
    probs = torch.sigmoid(logits)

    # Sample labels from Bernoulli using the computed probabilities
    costs = torch.sum(torch.tensor(action_costs) * A_one_hot_policy, dim=1)
    mean_rewards = probs - costs
    square_rewards = probs * (1-costs)**2 + (1-probs) * costs**2
    
    mean_reward = torch.mean(mean_rewards).item()
    mean_square_reward = torch.mean(square_rewards).item()

    # Compute the variance
    variance = mean_square_reward - mean_reward**2
    dro_reward = mean_reward - np.sqrt(delta * variance)

    # Return the mean reward
    return mean_reward, dro_reward


def true_evaluation_data(weights, bias, X, action_costs, A_one_hot_policy):
    """Evaluate the data generation process by generating synthetic data and computing statistics."""
    X_augmented = np.hstack([X, A_one_hot_policy])  # shape: (n_samples, 7)
    logits = X_augmented @ weights + bias
    probs = 1 / (1 + np.exp(-logits))

    costs = np.sum(action_costs * A_one_hot_policy, axis=1)
    rewards = probs - costs
    mean_reward = np.mean(rewards)
    # Return the mean reward
    return mean_reward

def true_evaluation_data_dro(weights, bias, X, action_costs, A_one_hot_policy, delta):
    """Evaluate the data generation process by generating synthetic data and computing statistics."""
    X_augmented = np.hstack([X, A_one_hot_policy])  # shape: (n_samples, 7)
    logits = X_augmented @ weights + bias
    probs = 1 / (1 + np.exp(-logits))

    costs = np.sum(action_costs * A_one_hot_policy, axis=1)
    rewards = probs - costs
    square_rewards = probs * (1-costs)**2 + (1-probs) * costs**2
    mean_reward = np.mean(rewards)
    mean_square_reward = np.mean(square_rewards)
    # Compute the variance
    variance = mean_square_reward - mean_reward**2
    dro_reward = mean_reward - np.sqrt(delta * variance)
    # Return the mean reward
    return mean_reward, dro_reward


def compute_objective(policy, X_pt, action_costs_pt, weights_pt, bias_pt):
    """Compute the mean reward for the given policy and the contexts."""
    A_one_hot_policy = policy(X_pt)
    # Create an augmented input for the logistic function
    X_augmented = torch.cat(
        (X_pt, A_one_hot_policy), dim=1
    )  # shape: (n_samples, n_features + n_actions)
    logits = torch.matmul(X_augmented, weights_pt) + bias_pt
    probs = torch.sigmoid(logits)

    rewards = probs - torch.sum(action_costs_pt * A_one_hot_policy, dim=1)
    mean_reward = torch.mean(rewards)

    return mean_reward



def optimise_policy(seed, action_costs, no_samples, generation_seed=0):
    """
    Optimize the policy using the generated data.

    Args:
        seed (int): Random seed for reproducibility.
        policy (str): Policy type for data generation.
        no_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing the optimized policy and computed statistics.
    """
    # Generate synthetic data - only gather contexts
    X, _, _, gen_dict = generate_data(seed, no_samples, generation_seed=generation_seed)

    # Apply the policy to the generated data
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # Compute the true rewards
    weights = torch.tensor(gen_dict["weights"], dtype=torch.float32)
    bias = torch.tensor(gen_dict["bias"], dtype=torch.float32)
    action_costs_pt = torch.tensor(action_costs, dtype=torch.float32)
    
    # Initialize the policy
    policy = NeuralNetworkPolicy(input_dim=X_tensor.shape[1], output_dim=5)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    for i in range(300):
        # Zero the gradients
        optimizer.zero_grad()

        # Compute the mean reward
        mean_reward = compute_objective(policy, X_tensor, action_costs_pt, weights, bias)

        # Compute the loss (negative mean reward)
        loss = -mean_reward

        # Backpropagation
        loss.backward()

        # Update the policy parameters
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}, Mean Reward: {mean_reward.item()}")

    # Compute the final mean reward
    mean_reward = true_evaluation(seed, policy, action_costs, no_samples, generation_seed=generation_seed)

    # Return the mean reward
    return mean_reward, policy


# if __name__ == "__main__":
#     # Example usage
#     seed = 10
#     no_samples = 10000
#     action_costs = np.array(
#         [0.25, 0.52, 0.63, 0.39, 0.57]
#     )

#     mean_reward, policy = optimise_policy(seed, action_costs, no_samples)

#     print(f"Optimized Mean Reward: {mean_reward}")

#     no_samples_arr = [
#         300,
#         500,
#         1000,
#         2000,
#         3000,
#         4000,
#         5000,
#         10000,
#         20000,
#         30000,
#         50000,
#         100000,
#         200000,
#         500000,
#     ]

#     policy_evaluations = []
#     for no_samples in no_samples_arr:
#         mean_reward = true_evaluation(seed, policy, action_costs, no_samples)
#         print(f"Mean Reward: {mean_reward}")
#         policy_evaluations.append(mean_reward)
#     # Plotting the results
#     import matplotlib.pyplot as plt
#     plt.plot(
#         no_samples_arr,
#         policy_evaluations,
#         marker="o",
#     )
#     plt.xlabel("Number of Samples")
#     plt.ylabel("Mean Reward")
#     plt.title("Mean Reward vs Number of Samples")
#     plt.xscale("log")
#     plt.grid()
#     plt.show()

#     # Save the policy
#     torch.save(policy.state_dict(), "policy.pth")
#     print("Policy saved to policy.pth")
#     # Load the policy
