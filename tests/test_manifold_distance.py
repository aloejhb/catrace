import pytest
import numpy as np
import pandas as pd

from catrace.manifold_distance import (
    compute_distances_df, DistanceConfig, ShuffleConfig,
    compute_distances_mat, get_mean_dist_mat,
    compute_center_euclidean, compute_center_euclidean_distance_mat,
    invert_cov_mat, compute_mahals, compute_euclideans,
    get_manifold_pair_distance_df
)
                           

@pytest.fixture
def example_df():
    """
    Create a simple DataFrame with a multi-index containing 'time' and 'odor'.
    We'll generate synthetic data for 2 odors and 5 time points each, with 3 neurons.
    """
    n_timepoints = 5
    n_neurons = 3
    manifold_names = ['A', 'B']

    # Create multi-index:
    # time from 0..4 for odor 1, then time from 0..4 for odor 2
    arrays = [
        np.concatenate([np.arange(n_timepoints)]*len(manifold_names)),  # time
        np.repeat(manifold_names, n_timepoints)                         # odor
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=["time", "manifold"])
    
    # Random data for 3 neurons
    np.random.seed(42)
    data = np.random.rand(len(index), n_neurons)

    df = pd.DataFrame(data, index=index, columns=range(n_neurons))
    # Namve the level of column to 'neuron'
    df.columns.name = 'neuron'
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

    assert dist_df.index.names == ["sample_manifold", "ref_manifold"]


def test_compute_distances_df_euclidean(example_df):
    """
    Test compute_distances_df with Euclidean distances.
    """
    distance_cfg = DistanceConfig(metric='euclidean')
    dist_df = compute_distances_df(example_df, distance_cfg=distance_cfg)
    # Should have the same shape (4, 5) given 2 odors * 2 odors rows, 5 columns
    assert dist_df.shape == (4, 5)

    # Check some distances. For example, distances from manifold of odor1
    # to itself should be relatively small compared to odor1 -> odor2, etc.
    manifold1_to_manifold1 = dist_df.xs(('A', 'A'),
                                        level=['sample_manifold',
                                               'ref_manifold']).values
    manifold1_to_manifold2 = dist_df.xs(('A', 'B'),
                                        level=['sample_manifold',
                                               'ref_manifold']).values
    # On average, we expect odor1->odor1 distances to be smaller than odor1->odor2
    assert np.mean(manifold1_to_manifold1) < np.mean(manifold1_to_manifold2)


def test_invalid_metric():
    """
    Test that providing an invalid metric raises ValueError.
    """
    with pytest.raises(ValueError):
        distance_cfg = DistanceConfig(metric='invalid_metric')


def test_shuffle_options_mutual_exclusion(example_df):
    """
    Test that trying to enable both shuffle options raises an error.
    """
    with pytest.raises(ValueError):
        shuffle_cfg = ShuffleConfig(
            do_pair=True,
            do_global=True
        )


def test_shuffle_global(example_df):
    """
    Test that global shuffle rearranges the rows but keeps the shape.
    """
    distance_cfg = DistanceConfig(metric='euclidean')
    shuffle_cfg = ShuffleConfig(
        do_global=True,
        seed_value=123
    )
    # Without shuffle
    dist_df_no_shuffle = compute_distances_df(example_df, distance_cfg=distance_cfg)
    
    # With shuffle
    dist_df_shuffle = compute_distances_df(
        example_df,
        distance_cfg=distance_cfg,
        shuffle_cfg=shuffle_cfg
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
    distance_cfg = DistanceConfig(metric='euclidean')
    shuffle_cfg = ShuffleConfig(
        do_pair=True,
        seed_value=123
    )
    dist_df_no_shuffle = compute_distances_df(example_df, distance_cfg=distance_cfg)
    print("Example data:")
    print(example_df)
    print("Distances:")
    print(dist_df_no_shuffle)

    dist_df_shuffle = compute_distances_df(
        example_df, 
        distance_cfg=distance_cfg,
        shuffle_cfg=shuffle_cfg
    )
    print("Distances after shuffle:")
    print(dist_df_shuffle)

    # Manifold1 -> Manifold1 distances and Manifold2 -> Manifold2 distances 
    # should remain the same because we do not shuffle identical odors
    # Assert numpy arrays are equal
    assert np.array_equal(
        dist_df_no_shuffle.xs(('A', 'A'),
                              level=['sample_manifold', 'ref_manifold']).values,
        dist_df_shuffle.xs(('A', 'A'),
                           level=['ref_manifold', 'sample_manifold']).values
    )
    
    assert np.array_equal(
        dist_df_no_shuffle.xs(('B', 'B'),
                              level=['sample_manifold', 'ref_manifold']).values,
        dist_df_shuffle.xs(('B', 'B'),
                           level=['ref_manifold', 'sample_manifold']).values
    )

    # Manifold1 -> Manifold2 distances or Manifold2 -> Manifold1 
    # should change because we shuffle only if manifold1 != manifold2.
    assert not np.array_equal(
        dist_df_no_shuffle.xs(('A', 'B'),
            level=['sample_manifold', 'ref_manifold']).values,
        dist_df_shuffle.xs(('A', 'B'),
            level=['ref_manifold', 'sample_manifold']).values
    )

    assert not np.array_equal(
        dist_df_no_shuffle.xs(('B', 'A'),
            level=['sample_manifold', 'ref_manifold']).values,
        dist_df_shuffle.xs(('B', 'A'),
            level=['ref_manifold', 'sample_manifold']).values
    )


def test_get_mean_dist_mat(example_df):
    """
    Test get_mean_dist_mat for correct reshaping and averaging.
    """
    dist_df = compute_distances_df(example_df, distance_cfg=DistanceConfig(metric='euclidean'))
    ordered_labels = ['A', 'B']
    dist_mat = get_mean_dist_mat(dist_df, ordered_labels)

    assert isinstance(dist_mat, pd.DataFrame)
    assert dist_mat.shape == (2, 2)
    assert list(dist_mat.index) == ordered_labels
    assert list(dist_mat.columns) == ordered_labels
    assert dist_mat.loc['A', 'A'] == np.mean(get_manifold_pair_distance_df(dist_df,'A', 'A').values)

def test_compute_distances_mat(example_df):
    """
    Test compute_distances_mat which combines compute_distances_df and get_mean_dist_mat.
    """
    ordered_labels = ['A', 'B']
    dist_mat = compute_distances_mat(example_df, ordered_labels, distance_cfg=DistanceConfig(metric='euclidean'))

    assert isinstance(dist_mat, pd.DataFrame)
    assert dist_mat.shape == (2, 2)
    assert list(dist_mat.index) == ordered_labels
    assert list(dist_mat.columns) == ordered_labels

    # Check if diagonal elements are smaller (A->A vs A->B) for Euclidean
    # This depends on the data, but for the fixture it should hold.
    assert dist_mat.loc['A', 'A'] < dist_mat.loc['A', 'B']
    assert dist_mat.loc['B', 'B'] < dist_mat.loc['B', 'A']

def test_compute_center_euclidean(example_df):
    """
    Test compute_center_euclidean for calculating distance between manifold centers.
    """
    manifold_A = example_df.xs('A', level='manifold')
    manifold_B = example_df.xs('B', level='manifold')

    dist_A_A = compute_center_euclidean(manifold_A, manifold_A)
    dist_A_B = compute_center_euclidean(manifold_A, manifold_B)

    assert isinstance(dist_A_A, float)
    assert dist_A_A == 0.0, "Distance from a manifold center to itself should be 0."
    assert dist_A_B > 0.0, "Distance between different manifold centers should be positive."

def test_compute_center_euclidean_distance_mat_no_shuffle(example_df):
    """
    Test compute_center_euclidean_distance_mat without shuffling.
    """
    ordered_labels = ['A', 'B']
    # Assuming the bug in compute_center_euclidean_distance_mat regarding 'df' vs 'dff' is fixed
    # or this test will help identify it.
    dist_mat = compute_center_euclidean_distance_mat(example_df, ordered_labels)

    assert isinstance(dist_mat, pd.DataFrame)
    assert dist_mat.shape == (2, 2)
    assert list(dist_mat.index) == ordered_labels
    assert list(dist_mat.columns) == ordered_labels
    assert dist_mat.index.name == 'sample_manifold'
    assert dist_mat.columns.name == 'ref_manifold'
    assert dist_mat.loc['A', 'A'] == 0.0
    assert dist_mat.loc['B', 'B'] == 0.0
    assert dist_mat.loc['A', 'B'] > 0.0
    assert dist_mat.loc['A', 'B'] == dist_mat.loc['B', 'A'] # Should be symmetric without shuffle

def test_compute_center_euclidean_distance_mat_with_shuffle(example_df):
    """
    Test compute_center_euclidean_distance_mat with global shuffling.
    The function makes the matrix symmetric after shuffling.
    """
    ordered_labels = ['A', 'B']
    shuffle_cfg = ShuffleConfig(do_global=True, seed_value=42)
    
    # Assuming the bug fix in the source code for 'df' vs 'dff'
    dist_mat_shuffled = compute_center_euclidean_distance_mat(
        example_df,
        ordered_labels,
        shuffle_cfg=shuffle_cfg
    )
    dist_mat_no_shuffle = compute_center_euclidean_distance_mat(
        example_df,
        ordered_labels,
        shuffle_cfg=ShuffleConfig() # Default no-shuffle
    )

    assert dist_mat_shuffled.shape == (2,2)

    assert dist_mat_shuffled.loc['A', 'A'] == pytest.approx(0.0)
    assert dist_mat_shuffled.loc['B', 'B'] == pytest.approx(0.0)

    # Off-diagonal elements should differ from non-shuffled version.
    assert dist_mat_shuffled.loc['A', 'B'] != dist_mat_no_shuffle.loc['A', 'B']
    assert dist_mat_shuffled.loc['A', 'B'] == dist_mat_shuffled.loc['B', 'A'] # Due to symmetrization

## Optional: Basic direct tests for core utility functions ##

def test_invert_cov_mat_simple():
    """ Test invert_cov_mat with a simple invertible matrix. """
    cov_mat = np.array([[2.0, 1.0], [1.0, 2.0]])
    inv_cov = invert_cov_mat(cov_mat, reg=0.0) # No explicit reg for this simple case
    identity_approx = cov_mat @ inv_cov
    assert np.allclose(identity_approx, np.eye(2)), "Matrix times its inverse should be identity."

def test_compute_mahals_simple():
    """ Test compute_mahals with simple inputs. """
    points = np.array([[1, 0], [0, 1], [2,2]])
    ref = np.array([0, 0])
    # Using identity as inv_cov_mat, Mahalanobis should be same as Euclidean squared (sqrt thereof)
    inv_cov_mat = np.eye(2)
    mahals = compute_mahals(points, ref, inv_cov_mat)
    expected_euclideans = [np.sqrt(1), np.sqrt(1), np.sqrt(8)]
    assert np.allclose(mahals, expected_euclideans)

def test_compute_euclideans_simple():
    """ Test compute_euclideans with simple inputs. """
    points = np.array([[1, 0], [0, 1], [3,4]])
    ref = np.array([0, 0])
    euclideans = compute_euclideans(points, ref)
    expected = [1.0, 1.0, 5.0] # sqrt(3^2+4^2) = 5
    assert np.allclose(euclideans, expected)

def test_compute_distances_df_single_manifold(example_df):
    """Test compute_distances_df when only one manifold exists in the data."""
    df_single_manifold = example_df.xs('A', level='manifold', drop_level=False)
    dist_df = compute_distances_df(df_single_manifold)
    
    assert isinstance(dist_df, pd.DataFrame)
    # 1 manifold (A) -> 1x1 = 1 pair (A,A). 5 timepoints/points.
    assert dist_df.shape == (1, 5) 
    assert dist_df.index.names == ["sample_manifold", "ref_manifold"]
    assert dist_df.index[0] == ('A', 'A')

def test_compute_distances_df_empty_input():
    """Test compute_distances_df with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=['neuron1', 'neuron2'], 
                            index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=['time', 'manifold']))
    
    # Depending on implementation, this might raise an error or return an empty DataFrame.
    # Let's assume it should return an empty DataFrame gracefully.
    dist_df = compute_distances_df(empty_df)
    assert isinstance(dist_df, pd.DataFrame)
    assert dist_df.empty
    # If it's not empty, check its shape is (0,0) or (0, expected_columns_if_any)
    # For your current code, it will likely be (0,0) before any columns are determined.
    # Actually, it will have 0 rows, and columns based on `distances_dict.values()`.
    # If `distances_dict` is empty, `list(distances_dict.values())` is `[]`.
    # `pd.DataFrame([])` is an empty DataFrame with no columns.
    assert dist_df.shape == (0, 0) # Or more specifically (0,0) if no columns generated.
