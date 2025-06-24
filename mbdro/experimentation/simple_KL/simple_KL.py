import numpy as np
import matplotlib.pyplot as plt

# Increase the default font size for matplotlib
plt.rcParams.update({"font.size": 14})

def evaluate_robust_profit(policy, actions, delta):
    """
    Calculate the robust profit for a given policy, rewards, probabilities, and delta.

    Args:
        policy (np.ndarray): The policy to evaluate.
        rewards (np.ndarray): The rewards associated with the policy.
        probs (np.ndarray): The probabilities associated with the policy.
        delta (float): The delta value for robustness.

    Returns:
        float: The robust profit.
    """
    if delta ==0:
        # If delta is zero, return the expected profit
        return evaluate_profit(policy, actions)
    alpha_range = np.linspace(0.1, 100, 10000).reshape(-1, 1)
    rewards = []
    reward_probs = []
    for i in range(len(actions)):
        action_rewards = actions[i]
        for j in range(len(action_rewards["values"])):
            rewards.append(action_rewards["values"][j])
            reward_probs.append(action_rewards["probabilities"][j]*policy[i])
    rewards = np.array(rewards)/10
    reward_probs = np.array(reward_probs)

    print("Rewards: ", rewards)
    print("Reward Probs: ", reward_probs)

    exp_rewards = -rewards/alpha_range
    # print("Exp Rewards: ", exp_rewards)
    exp_rewards = np.exp(exp_rewards) * reward_probs
    # print("Exp Rewards: ", exp_rewards)
    exp_rewards = np.sum(exp_rewards, axis=1)
    # print("Sum Exp Rewards: ", exp_rewards)
    # Calculate the robust profit
    alpha_range = alpha_range.flatten()
    robust_profits = -alpha_range * np.log(exp_rewards) - alpha_range*delta

    # The robust profit is the maximum of the robust profits across all alpha values

    # plt.plot(alpha_range, robust_profits)
    # plt.xlabel("Alpha")
    # plt.ylabel("Robust Profit")
    # plt.title("Robust Profit vs Alpha")
    # plt.grid()
    # plt.show()
    robust_profit = np.max(robust_profits)
    # 

    return robust_profit*10

def evaluate_profit(policy, actions):
    rewards = []
    reward_probs = []
    for i in range(len(actions)):
        action_rewards = actions[i]
        for j in range(len(action_rewards["values"])):
            rewards.append(action_rewards["values"][j])
            reward_probs.append(action_rewards["probabilities"][j]*policy[i])
    rewards = np.array(rewards)
    reward_probs = np.array(reward_probs)
    return np.sum(rewards * reward_probs)


def find_optimal_policy(actions, delta):
    """
    Find the optimal policy for a given set of actions and delta.

    Args:
        actions (list): The actions to evaluate.
        delta (float): The delta value for robustness.

    Returns:
        np.ndarray: The optimal policy.
    """
    r_profits = []
    a1_prob_range = np.linspace(0., 1, 100)
    for a1_prob in a1_prob_range:
        # Create a policy with the given probabilities
        policy = np.array([a1_prob, 1-a1_prob])
        # Calculate the robust profit for the policy
        robust_profit_value = evaluate_robust_profit(policy, actions, delta)
        r_profits.append(robust_profit_value)
    r_profits = np.array(r_profits)
    # Find the index of the maximum robust profit
    max_index = np.argmax(r_profits)
    robust_profit = r_profits[max_index]
    # Get the corresponding policy
    optimal_a1_prob = a1_prob_range[max_index]
    optimal_policy = np.array([optimal_a1_prob, 1-optimal_a1_prob])

    # Evaluate the profit for the optimal policy
    profit = evaluate_profit(optimal_policy, actions)
    print("Optimal Policy: ", optimal_policy)
    print("Profit: ", profit)
    print("Robust Profit: ", robust_profit)
        
    return optimal_policy, profit, robust_profit


if __name__ == "__main__":
    policy = np.array([0.3, 0.7])
    actions = [
        {
            "values": [0, 1],
            "probabilities": [0.5, 0.5],
        },
        {
            "values": [-12, 2],
            "probabilities": [0.1, 0.9],
        },
    ]
    
    delta_arr = [0, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]

    standard_policy = np.array([0., 1.])

    robust_profits_robust = []
    robust_profits_standard = []
    standard_profits_robust = []
    standard_profits_standard = []


    for delta in delta_arr:
        print("Delta: ", delta)
        optimal_policy, profit, robust_profit = find_optimal_policy(actions, delta)
        robust_profits_robust.append(robust_profit)
        standard_profits_robust.append(profit)
        # Evaluate the roprofit for the standard policy
        standard_profit = evaluate_profit(standard_policy, actions)
        standard_profits_standard.append(standard_profit)
        robust_profit_standard = evaluate_robust_profit(standard_policy, actions, delta)
        robust_profits_standard.append(robust_profit_standard)

    # Transform the results to numpy arrays
    delta_arr = np.array(delta_arr)
    robust_profits_robust = np.array(robust_profits_robust).flatten()
    robust_profits_standard = np.array(robust_profits_standard).flatten()
    standard_profits_robust = np.array(standard_profits_robust).flatten()
    standard_profits_standard = np.array(standard_profits_standard).flatten()
    
    print("Delta: ", delta_arr)
    print("Robust Profits Robust policy: ", robust_profits_robust)
    print("Standard Profits Robust policy: ", standard_profits_robust)
    print("Standard Profits Standard policy: ", standard_profits_standard)
    print("Robust Profits Standard policy: ", robust_profits_standard)


    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.title("$V^{DRO}$ vs $\\delta$")
    plt.xlabel("$\\delta$")
    plt.ylabel("Profit")
    plt.plot(
        delta_arr,
        standard_profits_standard,
        "o-",
        label="$V^{s}(\\pi^{s})$",
    )
    plt.plot(
        delta_arr,
        standard_profits_robust,
        "x-",
        label="$V^s(\\pi^{DRO,\\delta})$" ,
    )
    plt.plot(
        delta_arr,
        robust_profits_standard,
        "^-",
        label="$V^{DRO,\\delta}(\\pi^{s})$",
    )
    plt.plot(
        delta_arr,
        robust_profits_robust,
        "s-",
        label="$V^{DRO,\\delta}(\\pi^{DRO,\\delta})$" ,
    )
    plt.legend()
    # Add grid lines with minor ticks
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig("robust_profit.png")
    plt.show()
    plt.close()
        