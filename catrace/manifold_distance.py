"""
Module: manifold_distance

This module provides functions for computing distances between neural activity
manifolds using Mahalanobis and Euclidean metrics. It includes utilities for 
data manipulation, shuffling, and visualization of distance matrices.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import catrace.process_time_trace as ptt

from scipy.spatial.distance import mahalanobis, euclidean

@dataclass
class DistanceConfig:
    """
    Configuration for distance computation.

    Args:
        metric: str, default='mahal'
            The distance metric to use. One of ['mahal', 'euclidean'].
        reg: float, default=1e-5
            Regularization term added to the covariance matrix for Mahalanobis distance.

    Raises:
        ValueError:
            If `metric` is not 'mahal' or 'euclidean'.
    """
    metric: str = 'mahal'
    reg: float = 1e-5
    def __post_init__(self):
        # Validation can be done directly in the config object
        if self.metric not in ['mahal', 'euclidean']:
            raise ValueError("Metric should be either 'mahal' or 'euclidean'.")


@dataclass
class ShuffleConfig:
    """
    Configuration for label shuffling during distance computation.

    Args:
        do_pair: bool, default=False
            Whether to shuffle labels between each pair of manifolds independently.
        do_global: bool, default=False
            Whether to shuffle all manifold labels globally across the dataset.
        seed_value: int or None, optional
            Seed for random number generation (for reproducibility).

    Raises:
        ValueError:
            If both `do_pair` and `do_global` are set to True.
    """
    do_pair: bool = False
    do_global: bool = False
    seed_value: Optional[int] = None

    def __post_init__(self):
        if self.do_pair and self.do_global:
            raise ValueError('Only one of do_pair or do_global can be True.')


def invert_cov_mat(cov_mat: np.ndarray,
                   reg: float=1e-5):
    """
    Invert a covariance matrix with regularization for numerical stability.

    Args:
        cov_mat: array-like, shape (n_neurons, n_neurons)
            The covariance matrix to be inverted.
        reg: float, default=1e-5
            Regularization term added to the covariance matrix for numerical stability.
    Returns:
        inv_cov_mat: array-like, shape (n_neurons, n_neurons)
            The inverted covariance matrix.
    """
    reg_term = reg * np.identity(cov_mat.shape[0])
    cov_mat += reg_term
    # Matrix Inversion using SVD for numerical stability
    u_mat, s_mat, v_mat = np.linalg.svd(cov_mat)
    inv_cov_mat = np.dot(v_mat.T, np.dot(np.diag(1 / s_mat), u_mat.T))
    return inv_cov_mat


def _compute_euclideans(points, ref):
    """
    Compute Euclidean distances

    Args:
        points: array-like, shape (n_points, n_neurons)
            The data points for which Euclidean distances are to be computed.
        ref: array-like, shape (n_neurons,)
            The reference point to which distances are computed.
    Returns:
        euclideans: array-like, shape (n_points,)
            The computed Euclidean distances for each point in `points`.
    """
    euclidean_distances = [euclidean(p, ref) for p in points]
    return euclidean_distances


def _compute_mahals(points, ref, inv_cov_mat):
    """
    Compute Mahalanobis distances

    Args:
        points: array-like, shape (n_points, n_neurons)
            The data points for which Mahalanobis distances are to be computed.
        ref: array-like, shape (n_neurons,)
            The reference point to which distances are computed.
        inv_cov_mat: array-like, shape (n_neurons, n_neurons)
            The inverse of the covariance matrix used in Mahalanobis distance computation.
    Returns:
        mahals: array-like, shape (n_points,)
            The computed Mahalanobis distances for each point in `points`.
    """
    mahals = [mahalanobis(p, ref, inv_cov_mat) for p in points]
    return mahals


def compute_pointwise_euclidean(manifold1: np.ndarray, manifold2: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances from each point in manifold1 to the center of manifold2.

    Args:
        manifold1: np.ndarray, shape (n_points, n_neurons)
            The data points to measure from.
        manifold2: np.ndarray, shape (m_points, n_neurons)
            The reference manifold whose center is used.

    Returns:
        distances: np.ndarray, shape (n_points,)
            Euclidean distances from each point in `manifold1` to the center of `manifold2`.
    """
    ref = manifold2.mean(axis=0)
    return _compute_euclideans(manifold1, ref)


def compute_pointwise_mahal(manifold1: np.ndarray, manifold2: np.ndarray, reg: float = 1e-5) -> np.ndarray:
    """
    Compute Mahalanobis distances from each point in manifold1 to the center of manifold2.

    Args:
        manifold1: np.ndarray, shape (n_points, n_neurons)
            The data points to measure from.
        manifold2: np.ndarray, shape (m_points, n_neurons)
            The reference manifold whose center is used.
        reg: float, default=1e-5
            Regularization for covariance matrix inversion.

    Returns:
        distances: np.ndarray, shape (n_points,)
            Mahalanobis distances from each point in `manifold1` to the center of `manifold2`.
    """
    ref = manifold2.mean(axis=0)
    cov = np.cov(manifold2, rowvar=False)
    inv_cov = invert_cov_mat(cov, reg=reg)
    return _compute_mahals(manifold1, ref, inv_cov)


def compute_mean_pointwise_distance(
    manifold1: np.ndarray, manifold2: np.ndarray, metric: str = 'euclidean', reg: float = 1e-5
) -> float:
    """
    Compute the mean pointwise distance between two manifolds.

    Args:
        manifold1: np.ndarray, shape (n_points, n_neurons)
            The data points to measure from.
        manifold2: np.ndarray, shape (m_points, n_neurons)
            The reference manifold whose center and covariance are used.
        metric: str, default='euclidean'
            The distance metric to use. One of ['mahal', 'euclidean'].
        reg: float, default=1e-5
            Regularization for covariance matrix inversion (only used for Mahalanobis distance).

    Returns:
        mean_distance: float
            The mean distance from points in `manifold1` to the center of `manifold2`.
    """
    if metric == 'mahal':
        distances = compute_pointwise_mahal(manifold1, manifold2, reg=reg)
    elif metric == 'euclidean':
        distances = compute_pointwise_euclidean(manifold1, manifold2)
    else:
        raise ValueError("Metric should be either 'mahal' or 'euclidean'.")
    return np.mean(distances)


def compute_distances_df(
    dff: pd.DataFrame,
    distance_config: DistanceConfig = DistanceConfig(),
    shuffle_config: ShuffleConfig = ShuffleConfig(),
    manifold_level: str = 'manifold',
) -> pd.DataFrame:
    """
    Compute distances between all pairs of manifolds in a DataFrame.

    Args:
        dff: pd.DataFrame
            A multi-index DataFrame with neural data. The index must include a
            'manifold' level, where each row corresponds to a data point in a manifold.
        distance_config: DistanceConfig
            Configuration for the distance metric and regularization.
        shuffle_config: ShuffleConfig
            Configuration for shuffling manifold labels before distance computation.

    Returns:
        distances_df: pd.DataFrame
            A DataFrame with shape (num_pairs, num_points). The index is a MultiIndex
            with levels ['sample_manifold', 'ref_manifold'], and each row contains
            distances from points in one manifold to the center of another.

    Raises:
        ValueError:
            If both pairwise and global shuffling are enabled.
            If an invalid distance metric is specified.
    """
    manifold_label_list = list(dff.index.unique(manifold_level))
    num_manifolds = len(manifold_label_list)

    if shuffle_config.do_global:
        dff = shuffle_manifold_labels_global(dff, seed=shuffle_config.seed_value)

    if shuffle_config.do_pair:
        if shuffle_config.seed_value is not None:
            master_rng = np.random.default_rng(shuffle_config.seed_value)
            seed_values = master_rng.integers(0, 1e9, size=num_manifolds ** 2).tolist()
        else:
            seed_values = [None] * (num_manifolds ** 2)

    distances_dict = dict()
    for label1 in manifold_label_list:
        for label2 in manifold_label_list:
            m1 = dff.xs(label1, level=manifold_level)
            m2 = dff.xs(label2, level=manifold_level)

            if shuffle_config.do_pair and label1 != label2:
                m1, m2 = shuffle_manifold_pair_labels(
                    m1, m2, seed_value=seed_values.pop(0)
                )

            center2 = m2.mean(axis=0)

            if distance_config.metric == 'mahal':
                inv_cov = invert_cov_mat(np.cov(m2.T), reg=distance_config.reg)
                distances = _compute_mahals(m1.to_numpy(), center2.to_numpy(), inv_cov)
            else:
                distances = _compute_euclideans(m1.to_numpy(), center2.to_numpy())

            distances_dict[(label1, label2)] = distances

    index = pd.MultiIndex.from_tuples(distances_dict.keys(),
                                    names=['sample_manifold', 'ref_manifold'])
    return pd.DataFrame(list(distances_dict.values()), index=index)


def shuffle_manifold_pair_labels(manifold1, manifold2, seed_value):
    """
    Shuffle the labels of two manifolds.
    Args:
        manifold1: pd.DataFrame. Rows are points and columns are neurons.
        manifold2: pd.DataFrame. Rows are points and columns are neurons.
        seed_value: int. Seed for random shuffling.
    Returns:
        manifold1: pd.DataFrame. Shuffled manifold1.
        manifold2: pd.DataFrame. Shuffled manifold2.
    """
    random_state = np.random.default_rng(seed_value)
    # Concatenate manifold1 and manifold2
    manifold1and2 = pd.concat([manifold1, manifold2])
    # Shuffle the rows
    manifold1and2_shuffled = manifold1and2.sample(
        frac=1, random_state=random_state
    )
    # Set the index to the original index
    manifold1and2_shuffled.index = manifold1and2.index
    # Split the shuffled manifold into two
    manifold1 = manifold1and2_shuffled.iloc[: len(manifold1)]
    manifold2 = manifold1and2_shuffled.iloc[len(manifold1) :]
    return manifold1, manifold2


def compute_distances_mat(dff, ordered_manifold_labels=None, **kwargs):
    """
    Compute distances between all pairs of manifolds in a DataFrame.
    This first computes the distances from each point in the sample manifold to the 
    reference manifold, and then averages the distances across all points in the sample manifold.
    Args:
        dff: pd.DataFrame
            A multi-index DataFrame with neural data. The index must include a
            'manifold' level, where each row corresponds to a data point in a manifold, and each column corresponds to a neuron.
        ordered_manifold_labels: list
            List of ordered manifold labels for the output distance matrix.
        kwargs: dict
            Additional arguments passed to the `compute_distances_df` function.
    """
    dist_df = compute_distances_df(dff, **kwargs)
    dist_mat = get_mean_dist_mat(dist_df, ordered_manifold_labels)
    return dist_mat


def compute_center_euclidean(manifold1, manifold2):
    """
    Compute euclidean distance between the centers of two manifolds
    Args:
        manifold1: pd.DataFrame. Rows are time points and columns are neurons.
        manifold2: pd.DataFrame. Rows are time points and columns are neurons.
    Returns:
        dist: float. Euclidean distance between the centers of the two manifolds.
    """
    center1 = manifold1.mean(axis=0)  # Average over time points
    center2 = manifold2.mean(axis=0)
    dist = euclidean(center1, center2)

    return dist


def compute_center_mahal(manifold1, manifold2, reg=1e-5):
    """
    Compute Mahalanobis distance from the center of manifold1 to manifold2.
    Args:
        manifold1: pd.DataFrame. Rows are time points and columns are neurons.
        manifold2: pd.DataFrame. Rows are time points and columns are neurons.
        reg: float, default=1e-5. Regularization term for covariance matrix inversion.
    Returns:
        dist: float. Mahalanobis distance between the centers of the two manifolds.
    """
    center1 = manifold1.mean(axis=0)  # Average over time points
    center2 = manifold2.mean(axis=0)
    cov = np.cov(manifold2.T)  # Covariance matrix of manifold2
    inv_cov = invert_cov_mat(cov, reg=reg)
    dist = mahalanobis(center1, center2, inv_cov)

    return dist


def compute_center_distance(
    manifold1, manifold2, metric='euclidean', reg=1e-5
):
    """
    Compute the distance between the centers of two manifolds.

    Args:
        manifold1: pd.DataFrame. Rows are time points and columns are neurons.
        manifold2: pd.DataFrame. Rows are time points and columns are neurons.
        metric: str, default='euclidean'. The distance metric to use. One of ['mahal', 'euclidean'].
        reg: float, default=1e-5. Regularization term for covariance matrix inversion (only used for Mahalanobis distance).

    Returns:
        dist: float. The distance between the centers of the two manifolds.
    """
    if metric == 'mahal':
        dist = compute_center_mahal(manifold1, manifold2, reg=reg)
    elif metric == 'euclidean':
        dist = compute_center_euclidean(manifold1, manifold2)
    else:
        raise ValueError("Metric should be either 'mahal' or 'euclidean'.")
    return dist


def shuffle_manifold_labels_global(dff, seed=None):
    """
    Shuffle the labels of the manifold globally.
    Args:
        dff: pd.DataFrame. Row index is a MultiIndex, where one of the levels is 'manifold'. The rows correspond to points in a given mainfold and the columns correspond to neurons.
        seed: int. Seed for random shuffling.
    Returns:
        df_shuffled: pd.DataFrame. Shuffled DataFrame with the same shape as dff.
    """
    random_state = np.random.default_rng(seed)
    # Shuffle the rows
    df_shuffled = dff.sample(frac=1, random_state=random_state)
    # Set the index to the original index
    df_shuffled.index = dff.index
    return df_shuffled


def compute_center_euclidean_distance_mat(
    dff,
    ordered_manifold_labels = None,
    shuffle_config: ShuffleConfig = ShuffleConfig(),
    manifold_level: str = 'manifold',
):
    """
    Compute the distance matrix between the centers of all pairs of manifolds.
    Args:
        dff: pd.DataFrame. Row index is a MultiIndex, where one of the levels is 'manifold'. The rows correspond to points in a given mainfold and the columns correspond to neurons.
        ordered_manifold_labels: list. List of ordered manifold labels for the output distance matrix.
        shuffle_config: ShuffleConfig. Configuration for shuffling manifold labels before distance computation.
    Returns:
        dist_mat: pd.DataFrame. A DataFrame with the distances between manifolds.
            The index and columns are the ordered manifold labels.
            The level name is 'sample_manifold' for the index and 'ref_manifold' for the columns, just to be consistent from the 'assymetric' distance matrix from the averaged point-to-manifold distances (`compute_distances_mat`).
    """
    if ordered_manifold_labels is None:
        ordered_manifold_labels = list(dff.index.unique(manifold_level))

    num_manifolds = len(ordered_manifold_labels)
    # Compute euclidean distance between centers
    dist_mat = pd.DataFrame(index=ordered_manifold_labels,
                            columns=ordered_manifold_labels,
                            dtype=float)
    # The level name is 'odor' for the index and 'ref_odor' for the columns
    dist_mat.index.name = 'sample_manifold'
    dist_mat.columns.name = 'ref_manifold'

    if shuffle_config.do_global:
        dff = shuffle_manifold_labels_global(dff, seed=shuffle_config.seed_value)

    if shuffle_config.do_pair:
        if shuffle_config.seed_value is not None:
            master_rng = np.random.default_rng(shuffle_config.seed_value)
            seed_values = master_rng.integers(
                0, 1e9, size=num_manifolds * num_manifolds
            ).tolist()
        else:
            seed_values = [None] * num_manifolds * num_manifolds

    for ml1 in ordered_manifold_labels:
        for ml2 in ordered_manifold_labels:
            manifold1 = dff.xs(ml1, level=manifold_level)
            manifold2 = dff.xs(ml2, level=manifold_level)
            if shuffle_config.do_pair and ml1 != ml2:
                manifold1, manifold2 = shuffle_manifold_pair_labels(
                    manifold1, manifold2, seed_value=seed_values.pop(0)
                )
            dist = compute_center_euclidean(manifold1, manifold2)
            dist_mat.loc[ml1, ml2] = dist

    # Make dist_mat symmetric, this is necessary
    # because shuffling introduces asymmetry from random sampling
    dist_mat = (dist_mat + dist_mat.T) / 2

    return dist_mat


def get_mean_dist_mat(dist_df, ordered_manifold_labels=None):
    """
    Compute the mean distance matrix from the distances DataFrame.

    Args:
        dist_df: pd.DataFrame
            A DataFrame representing point-to-manifold distances, computed by `compute_distances_df`.
            The index is a MultiIndex with levels ['sample_manifold', 
            'ref_manifold'], and each row contains distances from points in
            sample_manifold to the center of ref_manifold.
        ordered_manifold_labels: list, optional
            List of ordered manifold labels for the output distance matrix.

    Returns:
        dist_mat: pd.DataFrame
            A DataFrame with the distances between manifolds.
            The index and columns are the ordered manifold labels.
    """
    if ordered_manifold_labels is None:
        ordered_manifold_labels = list(dist_df.index.unique('sample_manifold'))
    dist_mean = dist_df.mean(axis=1)
    dist_mean.name = 'mean_dist'
    dist_mean = dist_mean.reset_index()

    dist_mean['order'] = range(len(dist_mean))
    dist_mat = dist_mean.pivot_table(
        index='sample_manifold', columns='ref_manifold', values='mean_dist'
    )

    dist_mat.index = pd.Categorical(
        dist_mat.index, categories=ordered_manifold_labels, ordered=True
    )
    dist_mat = dist_mat.sort_index()
    dist_mat = dist_mat[ordered_manifold_labels]
    return dist_mat


def get_manifold_pair_distance_df(dist_df,
                                  sample_manifold_label,
                                  ref_manifold_label):
    """
    Get the distance DataFrame for a specific pair of manifolds.
    Args:
        dist_df: pd.DataFrame
            A DataFrame representing point-to-manifold distances, computed by `compute_distances_df`.
        sample_manifold_label: str
            The label of the sample manifold.
        ref_manifold_label: str
            The label of the reference manifold.
    Returns:
        pair_dist_df: pd.DataFrame
            A DataFrame with the distances from points in the sample manifold to the center of the reference manifold.
    """
    pair_dist_df = dist_df.xs((sample_manifold_label, ref_manifold_label),
                                level=['sample_manifold', 'ref_manifold'])
    return pair_dist_df
