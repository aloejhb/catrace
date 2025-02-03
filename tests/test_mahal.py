import pytest
import numpy as np
import pandas as pd

from catrace.mahal import compute_distances_df  # Adjust import as needed

@pytest.fixture
def example_df():
    """
    Create a simple DataFrame with a multi-index containing 'time' and 'odor'.
    We'll generate synthetic data for 2 odors and 5 time points each, with 3 neurons.
    """
    n_timepoints = 5
    n_neurons = 3
    odors = [1, 2]

    # Create multi-index:
    # time from 0..4 for odor 1, then time from 0..4 for odor 2
    arrays = [
        np.concatenate([np.arange(n_timepoints)]*len(odors)),  # time
        np.repeat(odors, n_timepoints)                         # odor
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=["time", "odor"])
    
    # Random data for 3 neurons
    np.random.seed(42)
    data = np.random.rand(len(index), n_neurons)

    df = pd.DataFrame(data, index=index, columns=[f"neuron_{i}" for i in range(n_neurons)])
    return df

def test_compute_distances_df_basic(example_df):
    """
    Test basic functionality with default parameters (Mahalanobis distance).
    """
    dist_df = compute_distances_df(example_df)
    # Check that the returned object is a DataFrame
    assert isinstance(dist_df, pd.DataFrame)

    # We have 2 odors, so the MultiIndex should have 2 x 2 = 4 rows
    # Each row should correspond to distances for each timepoint of odor1
    # Since odor1 has 5 timepoints, the width of dist_df is 5 for each row
    # => shape should be (4, 5)
    assert dist_df.shape == (4, 5)

    # Check that the index is a MultiIndex with levels ['odor', 'ref_odor']
    assert dist_df.index.names == ["odor", "ref_odor"]

def test_compute_distances_df_euclidean(example_df):
    """
    Test compute_distances_df with Euclidean distances.
    """
    dist_df = compute_distances_df(example_df, metric='euclidean')
    # Should have the same shape (4, 5) given 2 odors * 2 odors rows, 5 columns
    assert dist_df.shape == (4, 5)

    # Check some distances. For example, distances from manifold of odor1
    # to itself should be relatively small compared to odor1 -> odor2, etc.
    odor1_to_odor1 = dist_df.loc[(1, 1)].values  # distances for each row in odor1
    odor1_to_odor2 = dist_df.loc[(1, 2)].values
    # On average, we expect odor1->odor1 distances to be smaller than odor1->odor2
    assert np.mean(odor1_to_odor1) < np.mean(odor1_to_odor2)

def test_invalid_metric(example_df):
    """
    Test that providing an invalid metric raises ValueError.
    """
    with pytest.raises(ValueError):
        compute_distances_df(example_df, metric='invalid_metric')

def test_shuffle_options_mutual_exclusion(example_df):
    """
    Test that trying to enable both shuffle options raises an error.
    """
    with pytest.raises(ValueError):
        compute_distances_df(
            example_df, 
            do_shuffle_manifold_pair_labels=True,
            do_shuffle_manifold_labels_global=True
        )

def test_shuffle_global(example_df):
    """
    Test that global shuffle rearranges the rows but keeps the shape.
    """
    # Without shuffle
    dist_df_no_shuffle = compute_distances_df(example_df, metric='euclidean')
    
    # With shuffle
    dist_df_shuffle = compute_distances_df(
        example_df, 
        metric='euclidean', 
        do_shuffle_manifold_labels_global=True,
        shuffle_seed_value=123
    )
    # The shape should be the same
    assert dist_df_no_shuffle.shape == dist_df_shuffle.shape
    # The actual values should differ in at least some entries
    # We won't check them all, but we'll check that they're not identical
    assert not dist_df_no_shuffle.equals(dist_df_shuffle)

def test_shuffle_pairwise(example_df):
    """
    Test that pairwise shuffle changes data for odor1 != odor2 pairs only.
    """
    dist_df_no_shuffle = compute_distances_df(example_df, metric='euclidean')
    print("Example data:")
    print(example_df)
    print("Distances:")
    print(dist_df_no_shuffle)

    dist_df_shuffle = compute_distances_df(
        example_df, 
        metric='euclidean', 
        do_shuffle_manifold_pair_labels=True,
        shuffle_seed_value=123
    )
    print("Distances after shuffle:")
    print(dist_df_shuffle)

    # Odor 1 -> Odor 1 distances and Odor 2 -> Odor 2 distances 
    # should remain the same because we do not shuffle identical odors
    pd.testing.assert_series_equal(
        dist_df_no_shuffle.loc[(1, 1)], dist_df_shuffle.loc[(1, 1)]
    )
    pd.testing.assert_series_equal(
        dist_df_no_shuffle.loc[(2, 2)], dist_df_shuffle.loc[(2, 2)]
    )

    # Odor 1 -> Odor 2 distances or Odor 2 -> Odor 1 
    # should change because we shuffle only if odor1 != odor2.
    assert not dist_df_no_shuffle.loc[(1, 2)].equals(dist_df_shuffle.loc[(1, 2)])
    assert not dist_df_no_shuffle.loc[(2, 1)].equals(dist_df_shuffle.loc[(2, 1)])
