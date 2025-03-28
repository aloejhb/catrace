import numpy as np
import pandas as pd


def generate_synthetic_manifolds(
    num_manifolds: int,
    num_points_per_manifold: int,
    num_neurons: int,
    center_dist: float = 5.0,
    noise_std_list: float = 1.0,
) -> pd.DataFrame:
    """
    Generates synthetic neural activity data for multiple manifolds.

    Args:
        num_manifolds: Number of distinct manifolds to generate.
        num_points_per_manifold: Number of data points (trials/time steps) per manifold.
        num_neurons: Number of neurons (dimensions) in the neural activity space.
        center_dist: Base distance between manifold centers.
        noise_std_list: Standard deviations for the noise added to each manifold.

    Returns:
        A pandas DataFrame with row repsenting data points across all manifolds,
        and columns representing individual neurons.
    """
    assert len(noise_std_list) == num_manifolds, (
        "Length of noise_std must match the number of manifolds."
    )
    # Generate distinct centers for each manifold
    manifold_centers = np.random.randn(num_manifolds, num_neurons) * center_dist

    data_points = np.empty((num_manifolds, num_points_per_manifold, num_neurons))
    for i in range(num_manifolds):
        # Generate data points around each manifold's center with some noise
        manifold_points = (
            manifold_centers[i]
            + np.random.randn(num_points_per_manifold, num_neurons) * noise_std_list[i]
        )
        data_points[i] = manifold_points

    # Concatenate data points from all manifolds
    all_data = np.reshape(data_points, (-1, num_neurons))

    manifold_df = pd.DataFrame(all_data)
    manifold_df['manifold'] = np.repeat(
        np.arange(num_manifolds), num_points_per_manifold
    )
    # Name the level of the column
    manifold_df.columns.name = 'neuron'
    manifold_df = manifold_df.set_index('manifold')
    return manifold_df