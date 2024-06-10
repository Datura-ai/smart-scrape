import numpy as np
import matplotlib.pyplot as plt
import torch

# Load the data from the file
with open("dict.py", "r") as file:
    data = eval(file.read())

# Extract UIDs and weights
processed_weight_uids = list(data.keys())
processed_weights = np.array(list(data.values()))


# Function to apply a more pronounced exponential exaggeration and normalize to sum to 1
def exponential_exaggeration_equal_weights(weights, factor=5):
    num_entries = len(weights)
    sorted_indices = np.argsort(weights)
    sorted_weights = weights[sorted_indices]

    # Calculate ranks with mean for duplicates
    unique_weights, inverse_indices, counts = np.unique(
        sorted_weights, return_inverse=True, return_counts=True
    )
    ranks = np.zeros_like(sorted_weights, dtype=float)
    for i, count in enumerate(counts):
        if count > 1:  # If there are duplicates
            # Assign the mean rank for all occurrences of this weight
            mean_rank = np.mean(np.arange(np.sum(counts[:i]), np.sum(counts[: i + 1])))
            ranks[inverse_indices == i] = mean_rank
        else:
            ranks[inverse_indices == i] = np.sum(counts[:i])

    # Apply the exponential exaggeration
    exaggerated_weights = np.exp((ranks / num_entries) * factor) - 1
    exaggerated_weights /= np.sum(exaggerated_weights)  # Normalize to sum to 1

    # Map back to original order
    result = np.zeros_like(weights)
    result[sorted_indices] = exaggerated_weights
    return result


# Apply the function
exaggerated_weights = exponential_exaggeration_equal_weights(processed_weights)
print("Processed Weights:", processed_weights)
print("Exaggerated Weights:", exaggerated_weights)
# Sort data by original weights
sorted_indices = np.argsort(processed_weights)
sorted_uids = np.array(processed_weight_uids)[sorted_indices]
sorted_weights = processed_weights[sorted_indices]
sorted_exaggerated_weights = exaggerated_weights[sorted_indices]

# Plot the values before and after
plt.figure(figsize=(14, 7))

# Use the indices of the sorted weights for the x-axis
indices = np.arange(len(sorted_weights))

plt.subplot(1, 2, 1)
plt.plot(indices, sorted_weights, label="Original Weights", color="blue")
plt.xlabel("Index (sorted by original weights)")
plt.ylabel("Weight")
plt.title("Original Weights")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(indices, sorted_exaggerated_weights, label="Exaggerated Weights", color="red")
plt.xlabel("Index (sorted by original weights)")
plt.ylabel("Exaggerated Weight")
plt.title("Exaggerated Weights (More Exponential and Normalized to Sum to 1)")
plt.grid(True)

plt.tight_layout()

# Save the plot as an image
plt.savefig("exaggerated_weights_comparison.png")
plt.show()


# Function to normalize weights (for integration with other systems)
def normalize_weights(processed_weight_uids, processed_weights, factor=5):
    # Convert to numpy array for processing
    weights = np.array([weight for weight in processed_weights])

    # # Sort weights and apply exponential exaggeration
    # def exponential_exaggeration(weights, factor=5):
    #     num_entries = len(weights)
    #     sorted_indices = np.argsort(weights)
    #     ranks = np.arange(num_entries)
    #     exaggerated_weights = np.exp((ranks / num_entries) * factor) - 1
    #     exaggerated_weights /= np.sum(exaggerated_weights)  # Normalize to sum to 1

    #     result = np.zeros_like(weights)
    #     result[sorted_indices] = exaggerated_weights
    #     return result

    exaggerated_weights = exponential_exaggeration_equal_weights(weights, factor)

    # Normalize weights to sum to 1
    normalized_weights = exaggerated_weights / np.sum(exaggerated_weights)

    # Convert back to torch tensor
    normalized_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)

    return normalized_weights_tensor


# Example usage of normalize_weights function
normalized_weights = normalize_weights(processed_weight_uids, processed_weights)
print("Normalized Weights:", normalized_weights)
