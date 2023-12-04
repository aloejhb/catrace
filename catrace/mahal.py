import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

def invert_cov_mat(cov_mat, reg=1e-5):
    reg_term = reg * np.identity(cov_mat.shape[0])
    cov_mat += reg_term
    # Matrix Inversion using SVD for numerical stability
    u, s, v = np.linalg.svd(cov_mat)
    inv_cov_mat = np.dot(v.T, np.dot(np.diag(1/s), u.T))
    return inv_cov_mat

def compute_mahals(points, ref, inv_cov_mat):
    mahals = [mahalanobis(p, ref, inv_cov_mat) for p in points]
    return mahals

def compute_mahals_df(df):
    odor_list = list(df.index.unique('odor'))

    # Compute center for each odor
    centers = df.groupby(level='odor').mean()

    # Compute cov_mat for each odor
    cov_mats = dict()
    for name, group in df.groupby(level='odor'):
        cov_mats[name] = np.cov(group.transpose())

    # Compute inv_cov_mat for each odor
    inv_cov_mats = dict()
    for key, val in cov_mats.items():
        inv_cov_mats[key] = invert_cov_mat(val, reg=1e-5)

    # For each ordered pair of odors,
    # compute mahal distance from all points of odor1 to center of odor2
    mahals_dict = dict()
    for odor1 in odor_list:
        data1 = df.xs(odor1, level='odor')
        for odor2 in odor_list:
            center2 = centers.loc[odor2]
            inv_cov_mat2 = inv_cov_mats[odor2]
            mahals_dict[(odor1, odor2)] = compute_mahals(data1.to_numpy(), center2.to_numpy(), inv_cov_mat2)



    # Convert to DataFrame with separate columns for odors
    multi_index = pd.MultiIndex.from_tuples(mahals_dict.keys(), names=['odor', 'ref_odor'])

    # Convert to DataFrame using MultiIndex
    mahals_df = pd.DataFrame(list(mahals_dict.values()), index=multi_index)
    return mahals_df
