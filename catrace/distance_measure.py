import itertools
import umap
import numpy as np
from sklearn.metrics.pairwise import paired_distances
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


def plot_distance_mat(dist_mat, odor_range, n_trials_per_odor, ax):
    same_dist = [get_same_odor_avgcorr(dist_mat, od, n_trials_per_odor, sigma=0)
                for od in odor_range]
    odor_pairs = itertools.combinations(odor_range, 2)
    diff_dist = [(odp, get_paired_odor_avgcorr(dist_mat, odp, n_trials_per_odor, sigma=0))
                for odp in odor_pairs]

    for i, sd in enumerate(same_dist):
        ax.plot(sd, label=f'#{i}')

    for odp, dd in diff_dist:
        ax.plot(dd, label=f'#{odp[0]} vs #{odp[0]}')

    ax.legend()
