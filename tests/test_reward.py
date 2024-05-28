import torch
import matplotlib.pyplot as plt
from neurons.validators.reward import BaseRewardModel
import numpy as np


def test_normalize_rewards():
    # Load the data from the file
    with open("dict.py", "r") as file:
        data = eval(file.read())

    # Extract values from the dictionary and convert them to a list
    rewards_list = list(data.values())

    # Convert the list of rewards to a torch tensor
    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    reward_model = BaseRewardModel()
    reward_model.var = (
        0  # Set variance to simulate condition where normalization is needed
    )

    # Normalize the rewards
    normalized_rewards = reward_model.normalize_rewards(rewards)
    sorted_rewards_list = sorted(rewards_list)
    sorted_normalized_rewards = torch.sort(normalized_rewards).values.numpy()
    print("Rewards List:", sorted_rewards_list)
    print("Normalized Rewards:", sorted_normalized_rewards)

    # Plot the original and normalized rewards for comparison
    plt.figure(figsize=(14, 7))

    # Sort the original rewards and obtain indices for sorting
    rewards_np = rewards.numpy() if isinstance(rewards, torch.Tensor) else rewards
    sorted_indices = np.argsort(rewards_np)
    sorted_rewards = rewards_np[sorted_indices]

    # Sort the normalized rewards using the same indices
    normalized_rewards_np = (
        normalized_rewards.numpy()
        if isinstance(normalized_rewards, torch.Tensor)
        else normalized_rewards
    )
    sorted_normalized_rewards = normalized_rewards_np[sorted_indices]

    # Generate indices for x-axis
    indices = np.arange(len(sorted_rewards))

    plt.subplot(1, 2, 1)
    plt.plot(indices, sorted_rewards, label="Original Rewards", color="blue")
    plt.xlabel("Index")
    plt.ylabel("Reward")
    plt.title("Original Rewards (Sorted)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        indices, sorted_normalized_rewards, label="Normalized Rewards", color="red"
    )
    plt.xlabel("Index")
    plt.ylabel("Normalized Reward")
    plt.title("Normalized Rewards (Sorted)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Assert conditions to verify normalization logic
    assert torch.all(
        normalized_rewards[rewards == 0] == 0
    ), "Zero rewards should remain zero after normalization."
    assert (
        torch.min(normalized_rewards) >= 0
    ), "Normalized rewards should not be negative."
    assert torch.max(normalized_rewards) <= 1, "Normalized rewards should not exceed 1."


# Call the test function
test_normalize_rewards()
