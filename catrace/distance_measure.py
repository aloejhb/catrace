import itertools
import umap
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import (paired_distances, euclidean_distances,
                                      cosine_distances)
from scipy.spatial.distance import pdist, squareform
from catrace.pattern_correlation import (
    get_same_odor_avgcorr,
    get_paired_odor_avgcorr)


def compute_distance_mat(embeddf, metric_name):
    umap_obj = umap.UMAP(output_metric=metric_name)
    metric = umap_obj.metric

    odor_idx = embeddf.index.unique('odor')
    trial_idx = embeddf.index.unique('trial')
    trials = list(itertools.product(odor_idx, trial_idx))

    n_trial = len(trials)
    n_time = len(embeddf.index.unique('time_bin'))
    dist_mat = np.zeros((n_time, n_trial, n_trial))
    for i,t1 in enumerate(trials):
        for j,t2 in enumerate(trials):
            tc1 = embeddf.xs(t1, level=('odor', 'trial')).to_numpy()
            tc2 = embeddf.xs(t2, level=('odor', 'trial')).to_numpy()
            dist_mat[:, i, j] = paired_distances(tc1, tc2, metric=metric)
    return dist_mat


def plot_distance_mat(dist_mat, odor_range, n_trials_per_odor, ax=None):
    same_dist = [get_same_odor_avgcorr(dist_mat, od, n_trials_per_odor, sigma=0)
                for od in odor_range]
    odor_pairs = itertools.combinations(odor_range, 2)
    diff_dist = [(odp, get_paired_odor_avgcorr(dist_mat, odp, n_trials_per_odor, sigma=0))
                for odp in odor_pairs]

    for i, sd in enumerate(same_dist):
        ax.plot(sd, label=f'#{i}')

    for odp, dd in diff_dist:
        ax.plot(dd, label=f'#{odp[0]} vs #{odp[1]}')

    ax.legend()


def compute_distances_to_starting_point(embeddf):
    first_bin = embeddf.index.unique('time_bin')[0]
    start_point = embeddf.xs(first_bin,
                             level='time_bin').mean(axis=0).values.reshape(1,-1)

    distance_list = []
    for i, row in embeddf.iterrows():
        distance = euclidean_distances(start_point, row.values.reshape(1,-1))
        distance_list.append(distance[0][0])

    distances = pd.DataFrame(distance_list, index=embeddf.index)
    return distances


def compute_pairwise_distance(df, metric='cosine'):
    """
    Args:
    df: columns features, rows samples
    """
    dist = pdist(df, metric=metric)
    dist_matrix = squareform(dist)

    # set lower triangle and diagonal of distance matrix to NaN to eliminate symmetric entries
    idx = np.triu_indices(dist_matrix.shape[0], k=0)
    dist_matrix[idx] = np.nan

    idxs = df.index.get_level_values('odor')
    df_flat = pd.DataFrame(dist_matrix, columns=idxs, index=idxs)
    df_flat = df_flat.rename_axis('odor1', axis=0).rename_axis('odor2', axis=1).stack().reset_index()
    return df_flat
