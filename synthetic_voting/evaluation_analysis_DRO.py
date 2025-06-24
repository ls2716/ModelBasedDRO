from policy import NeuralNetworkPolicy
import torch
import pandas as pd
import numpy as np

from generate_data import generate_data
from learn_conversion import train_model
from true_evaluation import true_evaluation_dro, true_evaluation_data_dro


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


if __name__ == "__main__":
    # Set the random seed for reproducibility
    seed = 10
    delta = 0.1

    # Get the weigths
    _, _, _, gen_dict = generate_data(parameter_seed=seed, n_samples=10, verbose=False)
    weights = gen_dict["weights"]
    bias = gen_dict["bias"]

    action_costs = np.array([0.25, 0.52, 0.63, 0.39, 0.57])

    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # Load the policy model
    input_dim = 2
    output_dim = 5
    policy = NeuralNetworkPolicy(input_dim, output_dim)
    # Load the policy
    policy.load_state_dict(torch.load("policy.pth"))

    model_rewards_run = []
    data_rewards_run = []
    true_rewards_run = []
    no_sims = 200

    n_samples_arr = [
        300,
        500,
        1000,
        2000,
        3000,
        5000,
        20000,
        50000,
        100000,
        200000,
        500000,
    ]

    for i in range(no_sims):
        print(f"Simulation {i + 1}/{no_sims}")
        # Generate random data
        model_rewards = []
        data_rewards = []
        true_rewards = []

        for n_samples in n_samples_arr:
            X, A_one_hot, labels, gen_dict = generate_data(
                parameter_seed=seed, n_samples=n_samples, verbose=False
            )

            # Evaluate the policy using model evaluation
            actions_one_hot = apply_policy(X, policy)

            mean_reward_model, dro_reward_model = evaluate_policy_model_dro(
                X, A_one_hot, labels, action_costs, actions_one_hot, delta
            )
            # print(
            #     f"Mean reward from model evaluation: {mean_reward_model}, n_samples: {n_samples}"
            # )
            # Evaluate the policy using data evaluation
            mean_reward_data, dro_reward_data = evaluate_policy_data_dro(
                X, A_one_hot, labels, action_costs, actions_one_hot, delta
            )

            # print(
            #     f"Mean reward from data evaluation: {mean_reward_data}, n_samples: {n_samples}"
            # )

            # Evaluate the policy using true evaluation
            mean_reward_true, dro_reward_true = true_evaluation_data_dro(
                weights, bias, X, action_costs, actions_one_hot, delta
            )
            model_rewards.append(dro_reward_model)
            data_rewards.append(dro_reward_data)
            true_rewards.append(dro_reward_true)
        model_rewards_run.append(model_rewards)
        data_rewards_run.append(data_rewards)
        true_rewards_run.append(true_rewards)
    # Convert to numpy arrays
    model_rewards = np.array(model_rewards_run)
    data_rewards = np.array(data_rewards_run)
    true_rewards = np.array(true_rewards_run)

    # Evaluate the policy using true evaluation
    mean_reward_true, dro_reward_true = true_evaluation_dro(
        seed, policy, action_costs, no_samples=1000000, delta=delta
    )
    print(f"Mean reward from true evaluation: {dro_reward_true}")

    # Compute the absolute difference between the model and data evaluation and the true evaluation
    model_rewards_err = np.abs(model_rewards - dro_reward_true)
    data_rewards_err = np.abs(data_rewards - dro_reward_true)
    true_rewards_err = np.abs(true_rewards - dro_reward_true)

    # Compte the mean of the absolute difference
    model_rewards_err_mean = np.mean(model_rewards_err, axis=0)
    data_rewards_err_mean = np.mean(data_rewards_err, axis=0)
    true_rewards_err_mean = np.mean(true_rewards_err, axis=0)

    # Compute the standard deviation of the absolute difference
    model_rewards_err_std = np.std(model_rewards_err, axis=0)
    data_rewards_err_std = np.std(data_rewards_err, axis=0)
    true_rewards_err_std = np.std(true_rewards_err, axis=0)

    # Plot the aboslute difference between the model and data evaluation and the true evaluation
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.semilogx(
        n_samples_arr,
        model_rewards_err_mean,
        label="Model Evaluation",
    )
    plt.fill_between(
        n_samples_arr,
        model_rewards_err_mean - model_rewards_err_std,
        model_rewards_err_mean + model_rewards_err_std,
        alpha=0.2,
    )
    plt.semilogx(
        n_samples_arr,
        data_rewards_err_mean,
        label="Data Evaluation",
    )
    plt.fill_between(
        n_samples_arr,
        data_rewards_err_mean - data_rewards_err_std,
        data_rewards_err_mean + data_rewards_err_std,
        alpha=0.2,
    )
    plt.semilogx(
        n_samples_arr,
        true_rewards_err_mean,
        label="True Evaluation Data",
    )
    plt.fill_between(
        n_samples_arr,
        true_rewards_err_mean - true_rewards_err_std,
        true_rewards_err_mean + true_rewards_err_std,
        alpha=0.2,
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("Absolute Difference from True DRO Evaluation ")
    plt.title("Absolute Difference from True DRO Evaluation")
    plt.legend()
    plt.grid()
    plt.savefig("evaluation_analysis_err_dro.png")
    plt.show()


        # Plot the convergence of means and add std fill
    model_rewards_mean = np.mean(model_rewards, axis=0)
    model_rewards_std = np.std(model_rewards, axis=0)
    data_rewards_mean = np.mean(data_rewards, axis=0)
    data_rewards_std = np.std(data_rewards, axis=0)
    true_rewards_mean = np.mean(true_rewards, axis=0)
    true_rewards_std = np.std(true_rewards, axis=0)
    plt.figure(figsize=(10, 5))
    plt.semilogx(
        n_samples_arr,
        model_rewards_mean,
        label="Model Evaluation",
    )
    plt.fill_between(
        n_samples_arr,
        model_rewards_mean - model_rewards_std,
        model_rewards_mean + model_rewards_std,
        alpha=0.2,
    )
    plt.semilogx(
        n_samples_arr,
        data_rewards_mean,
        label="Data Evaluation",
    )
    plt.fill_between(
        n_samples_arr,
        data_rewards_mean - data_rewards_std,
        data_rewards_mean + data_rewards_std,
        alpha=0.2,
    )
    plt.semilogx(
        n_samples_arr,
        true_rewards_mean,
        label="True Evaluation Data",
    )
    plt.fill_between(
        n_samples_arr,
        true_rewards_mean - true_rewards_std,
        true_rewards_mean + true_rewards_std,
        alpha=0.2,
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("Mean DRO Reward")
    plt.title("Mean DRO Reward vs Number of Samples")
    plt.legend()
    plt.grid()
    plt.savefig("evaluation_analysis_mean_dro.png")
    plt.show()

