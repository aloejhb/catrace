# %% [markdown]
# # Manifold Distance Computation using `mahal`
#
# This notebook demonstrates how to use the `mahal` module to compute distances between neural activity manifolds. The module provides functionalities for computing distances using both Mahalanobis and Euclidean metrics, with options for regularization and label shuffling.
#
# We will cover:
# 1.  Generating synthetic neural data to represent manifolds.
# 2.  Computing distances between a single pair of manifolds.
# 3.  Computing distances between multiple manifolds.
# 4.  Exploring different distance metrics (Mahalanobis vs. Euclidean).
# 5.  Understanding the effect of regularization on Mahalanobis distance.
# 6.  Demonstrating label shuffling for statistical comparisons.
# 7.  Visualizing the distance matrices.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import all functions and classes from the mahal module
from catrace.manifold_distance import (
    DistanceConfig,
    ShuffleConfig,
    compute_mahals,
    compute_euclideans,
    invert_cov_mat,
    compute_distances_df,
    compute_distances_mat,
    compute_center_euclidean,
    compute_center_euclidean_distance_mat,
    shuffle_manifold_labels_global,
    shuffle_manifold_pair_labels,
    get_mean_dist_mat,
    get_manifold_pair_distance_df,
)

# Set a random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## 1. Generating Synthetic Neural Data
#
# We'll create a synthetic dataset representing neural activity from different "manifolds" (e.g., corresponding to different conditions or stimuli). Each manifold will be a collection of data points (neural responses across neurons).

# %%
def generate_manifold_data(
    num_manifolds: int,
    num_points_per_manifold: int,
    num_neurons: int,
    base_mean_dist: float = 5.0,
    noise_std: float = 1.0,
) -> pd.DataFrame:
    """
    Generates synthetic neural activity data for multiple manifolds.

    Args:
        num_manifolds: Number of distinct manifolds to generate.
        num_points_per_manifold: Number of data points (trials/time steps) per manifold.
        num_neurons: Number of neurons (dimensions) in the neural activity space.
        base_mean_dist: Base distance between manifold centers.
        noise_std: Standard deviation of the noise within each manifold.

    Returns:
        A pandas DataFrame with a MultiIndex ('manifold', 'point_idx')
        and columns representing individual neurons.
    """
    all_data = []
    manifold_labels = [f"Manifold_{i+1}" for i in range(num_manifolds)]

    # Generate distinct centers for each manifold
    manifold_centers = np.random.randn(num_manifolds, num_neurons) * base_mean_dist

    for i, manifold_label in enumerate(manifold_labels):
        # Generate data points around each manifold's center with some noise
        data_points = (
            manifold_centers[i]
            + np.random.randn(num_points_per_manifold, num_neurons) * noise_std
        )
        for j in range(num_points_per_manifold):
            all_data.append(
                {"manifold": manifold_label, "point_idx": j, **{f"neuron_{k}": val for k, val in enumerate(data_points[j])}}
            )

    df = pd.DataFrame(all_data)
    df = df.set_index(["manifold", "point_idx"])
    return df

# Generate some sample data
num_manifolds = 4
num_points_per_manifold = 50
num_neurons = 10
dff = generate_manifold_data(num_manifolds, num_points_per_manifold, num_neurons)
print("Sample Data Head:")
print(dff.head())
print("\nManifold Labels:", dff.index.unique('manifold').tolist())
print(f"Data shape: {dff.shape}")

# %% [markdown]
# ## 2. Computing Distances Between a Pair of Manifolds
#
# Let's start by looking at the distance between two specific manifolds. We can extract their data and use the low-level `compute_mahals` or `compute_euclideans` functions.

# %%
# Extract data for two manifolds
manifold_1_label = 'Manifold_1'
manifold_2_label = 'Manifold_2'

m1_data = dff.xs(manifold_1_label, level='manifold')
m2_data = dff.xs(manifold_2_label, level='manifold')

print(f"\nShape of {manifold_1_label} data: {m1_data.shape}")
print(f"Shape of {manifold_2_label} data: {m2_data.shape}")

# Calculate the center of the reference manifold (Manifold_2)
center_m2 = m2_data.mean(axis=0).to_numpy()
print(f"\nCenter of {manifold_2_label}:\n{center_m2[:5]}...") # Display first 5 elements

# Compute Euclidean distances from points in Manifold_1 to the center of Manifold_2
euclidean_dists_1_to_2 = compute_euclideans(m1_data.to_numpy(), center_m2)
print(f"\nEuclidean distances from {manifold_1_label} points to {manifold_2_label} center (first 5):")
print(euclidean_dists_1_to_2[:5])
print(f"Mean Euclidean distance: {np.mean(euclidean_dists_1_to_2):.2f}")


# Compute Mahalanobis distances from points in Manifold_1 to the center of Manifold_2
# We need the inverse covariance matrix of the reference manifold (Manifold_2)
cov_m2 = np.cov(m2_data.T)
inv_cov_m2 = invert_cov_mat(cov_m2, reg=1e-5) # Add a small regularization for stability

mahalanobis_dists_1_to_2 = compute_mahals(m1_data.to_numpy(), center_m2, inv_cov_m2)
print(f"\nMahalanobis distances from {manifold_1_label} points to {manifold_2_label} center (first 5):")
print(mahalanobis_dists_1_to_2[:5])
print(f"Mean Mahalanobis distance: {np.mean(mahalanobis_dists_1_to_2):.2f}")

# %% [markdown]
# ### Center-to-Center Euclidean Distance
# The module also provides a direct function to compute the Euclidean distance between the centers of two manifolds.

# %%
center_m1 = m1_data.mean(axis=0)
center_euclidean_dist = compute_center_euclidean(m1_data, m2_data)
print(f"Euclidean distance between the centers of {manifold_1_label} and {manifold_2_label}: {center_euclidean_dist:.2f}")

# %% [markdown]
# ## 3. Computing Distances Between Multiple Manifolds (Point-to-Manifold)
#
# The `compute_distances_df` function allows computing distances from all points in one manifold to the center of another, for all possible pairs of manifolds. This returns a DataFrame where each row contains the distances of individual points from a 'sample_manifold' to a 'ref_manifold' center.
#
# The `compute_distances_mat` then takes this `dist_df` and averages the distances for each pair, returning a square distance matrix.

# %%
# Get all manifold labels in order
ordered_manifold_labels = dff.index.unique('manifold').tolist()
print("Ordered Manifold Labels:", ordered_manifold_labels)

# Compute Mahalanobis distances for all pairs of manifolds
print("\nComputing Mahalanobis distances (point-to-manifold)...")
mahal_dist_df = compute_distances_df(dff, distance_cfg=DistanceConfig(metric='mahal', reg=0.1))
print("\nRaw Mahalanobis Distances DataFrame (first 5 rows):")
print(mahal_dist_df.head())

# Get the mean distance matrix from the raw distance DataFrame
mahal_dist_mat = get_mean_dist_mat(mahal_dist_df, ordered_manifold_labels)
print("\nMean Mahalanobis Distance Matrix:")
print(mahal_dist_mat)

# Visualize the Mahalanobis distance matrix
plt.figure(figsize=(7, 6))
sns.heatmap(mahal_dist_mat, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Mean Mahalanobis Distance Matrix (Point-to-Manifold)')
plt.xlabel('Reference Manifold')
plt.ylabel('Sample Manifold')
plt.show()

# Compute Euclidean distances for all pairs of manifolds
print("\nComputing Euclidean distances (point-to-manifold)...")
euclidean_dist_df = compute_distances_df(dff, distance_cfg=DistanceConfig(metric='euclidean'))
euclidean_dist_mat = get_mean_dist_mat(euclidean_dist_df, ordered_manifold_labels)
print("\nMean Euclidean Distance Matrix:")
print(euclidean_dist_mat)

# Visualize the Euclidean distance matrix
plt.figure(figsize=(7, 6))
sns.heatmap(euclidean_dist_mat, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Mean Euclidean Distance Matrix (Point-to-Manifold)')
plt.xlabel('Reference Manifold')
plt.ylabel('Sample Manifold')
plt.show()

# %% [markdown]
# ## 4. Computing Distances Between Multiple Manifolds (Center-to-Center)
#
# The `compute_center_euclidean_distance_mat` directly computes the Euclidean distance between the centroids of all manifold pairs, returning a symmetric distance matrix.

# %%
print("\nComputing Center-to-Center Euclidean Distance Matrix...")
center_euclid_mat = compute_center_euclidean_distance_mat(dff, ordered_manifold_labels)
print("\nCenter-to-Center Euclidean Distance Matrix:")
print(center_euclid_mat)

# Visualize the center-to-center Euclidean distance matrix
plt.figure(figsize=(7, 6))
sns.heatmap(center_euclid_mat, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Center-to-Center Euclidean Distance Matrix')
plt.xlabel('Reference Manifold')
plt.ylabel('Sample Manifold')
plt.show()

# %% [markdown]
# ## 5. Exploring Regularization in Mahalanobis Distance
#
# The `reg` parameter in `DistanceConfig` adds a regularization term to the covariance matrix, which can help with numerical stability, especially when the covariance matrix is singular or ill-conditioned (e.g., when `num_points_per_manifold` is less than `num_neurons`). A higher `reg` value tends to make the Mahalanobis distance more similar to Euclidean distance.

# %%
# Let's demonstrate with different regularization values
reg_values = [0.0, 0.5, 5.0]

plt.figure(figsize=(15, 5))
for i, reg_val in enumerate(reg_values):
    mahal_dist_mat_reg = compute_distances_mat(dff, ordered_manifold_labels,
                                               distance_cfg=DistanceConfig(metric='mahal', reg=reg_val))
    plt.subplot(1, len(reg_values), i + 1)
    sns.heatmap(mahal_dist_mat_reg, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title(f'Mahalanobis Distance (reg={reg_val})')
    plt.xlabel('Reference Manifold')
    plt.ylabel('Sample Manifold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Demonstrating Label Shuffling
#
# Shuffling manifold labels is crucial for generating null distributions for statistical testing (e.g., to determine if observed manifold distances are significantly different from chance). The `ShuffleConfig` class allows controlling how labels are shuffled.
#
# * `do_global=True`: Shuffles all manifold labels globally across the entire dataset. This maintains the internal structure of each manifold but assigns them to random labels.
# * `do_pair=True`: Shuffles labels within each pair of manifolds independently. This mixes data points between the two manifolds being compared.
#
# **Important:** Only one of `do_global` or `do_pair` can be `True` at a time.

# %%
# Re-generate data with a slight overlap for clearer shuffling effect
dff_shuffling = generate_manifold_data(num_manifolds=3, num_points_per_manifold=100, num_neurons=5,
                                       base_mean_dist=2.0, noise_std=1.5)
ordered_manifold_labels_shuf = dff_shuffling.index.unique('manifold').tolist()

print("\n--- Original (No Shuffling) Mahalanobis Distance ---")
mahal_dist_mat_orig = compute_distances_mat(dff_shuffling, ordered_manifold_labels_shuf,
                                            distance_cfg=DistanceConfig(metric='mahal', reg=0.1))
print(mahal_dist_mat_orig)

# Visualize original distance matrix
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.heatmap(mahal_dist_mat_orig, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Original Mahalanobis Distance')
plt.xlabel('Reference Manifold')
plt.ylabel('Sample Manifold')


# Global Shuffling
print("\n--- Global Shuffling (Mahalanobis) ---")
mahal_dist_mat_global_shuffled = compute_distances_mat(dff_shuffling, ordered_manifold_labels_shuf,
                                                      distance_cfg=DistanceConfig(metric='mahal', reg=0.1),
                                                      shuffle_cfg=ShuffleConfig(do_global=True, seed_value=123))
print(mahal_dist_mat_global_shuffled)

plt.subplot(1, 2, 2)
sns.heatmap(mahal_dist_mat_global_shuffled, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Mahalanobis Distance (Global Shuffling)')
plt.xlabel('Reference Manifold')
plt.ylabel('Sample Manifold')
plt.tight_layout()
plt.show()

# Pairwise Shuffling for point-to-manifold distances
print("\n--- Pairwise Shuffling (Mahalanobis) ---")
# For pairwise shuffling, off-diagonal elements should show more similar distances.
# Diagonal elements represent distances within the same shuffled manifold, so they might not be directly comparable to original.
mahal_dist_mat_pair_shuffled = compute_distances_mat(dff_shuffling, ordered_manifold_labels_shuf,
                                                     distance_cfg=DistanceConfig(metric='mahal', reg=0.1),
                                                     shuffle_cfg=ShuffleConfig(do_pair=True, seed_value=456))
print(mahal_dist_mat_pair_shuffled)

plt.figure(figsize=(7, 6))
sns.heatmap(mahal_dist_mat_pair_shuffled, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Mahalanobis Distance (Pairwise Shuffling)')
plt.xlabel('Reference Manifold')
plt.ylabel('Sample Manifold')
plt.show()

# Pairwise Shuffling for center-to-center Euclidean distances
print("\n--- Pairwise Shuffling (Center-to-Center Euclidean) ---")
center_euclid_mat_pair_shuffled = compute_center_euclidean_distance_mat(dff_shuffling, ordered_manifold_labels_shuf,
                                                                         shuffle_cfg=ShuffleConfig(do_pair=True, seed_value=789))
print(center_euclid_mat_pair_shuffled)

plt.figure(figsize=(7, 6))
sns.heatmap(center_euclid_mat_pair_shuffled, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Center Euclidean Distance (Pairwise Shuffling)')
plt.xlabel('Reference Manifold')
plt.ylabel('Sample Manifold')
plt.show()

# %% [markdown]
# ## 7. Retrieving Distances for a Specific Manifold Pair
#
# You can retrieve the original point-to-manifold distances for a specific pair using `get_manifold_pair_distance_df`. This is useful for further analysis or plotting distributions of distances.

# %%
# Let's get the raw distances from Manifold_1 to Manifold_3's center
sample_m_label = 'Manifold_1'
ref_m_label = 'Manifold_3'

pair_dist_df = get_manifold_pair_distance_df(mahal_dist_df, sample_m_label, ref_m_label)
print(f"\nRaw Mahalanobis distances from points in {sample_m_label} to center of {ref_m_label}:")
print(pair_dist_df.head())
print(f"Mean distance for this pair: {pair_dist_df.mean().values[0]:.2f}")

# Visualize the distribution of distances for this pair
plt.figure(figsize=(6, 4))
sns.histplot(pair_dist_df.iloc[:, 0], kde=True, bins=20)
plt.title(f'Distribution of Mahalanobis Distances\n({sample_m_label} points to {ref_m_label} center)')
plt.xlabel('Mahalanobis Distance')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# This notebook provides a comprehensive overview of the `mahal` module's capabilities for computing and analyzing distances between neural activity manifolds.