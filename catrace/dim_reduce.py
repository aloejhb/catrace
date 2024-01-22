import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import cross_val_score
import umap

from .scale import standard_scale
from .process_time_trace import select_odors_df, select_time_points


def plot_embed(embeddf):
    groups = embeddf.groupby(['odor'])
    if 'z' in embeddf.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for name, group in groups:
            ax.plot(group.x, group.y, zs=group.z, marker='o', linestyle='-', ms=4, label=name, alpha=0.7)
    else:
        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='-', ms=4, label=name, alpha=0.7)
    ax.legend()
    return fig


def plot_embed_trial(embeddf):
    odor_list = embeddf['odor'].unique()
    clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for trial in embeddf:
        cidx = odor_list.index(trial.odor)
        ax.plot(trial.x, trial.y, marker='o', linestyle='-', ms=4, label=name, alpha=0.7, color=clr_cycle[cidx])
    pass


def plot_embed_timecourse_all(embeddf, odor_list, select_odor):
    groups = embeddf.groupby(['odor'])
    clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        if name in select_odor:
            cidx = odor_list.index(name)
            points = plot_embed_timecourse(ax, group, name, clr_cycle[cidx])
    ax.legend()


def plot_embed_timecourse(ax, group, name, color):
    total_t = len(group.index.unique('time'))
    points = [ax.scatter(row['x'], row['y'], color=color, marker='o', alpha=index[2]/total_t) for index, row in group.iterrows()]
    points[0].label = name
    return points


# def svd(X):
#   # Compute full SVD
#   U, Sigma, Vh = np.linalg.svd(X,
#       full_matrices=False, # It's not necessary to compute the full matrix of U or V
#       compute_uv=True)
  # X_svd = np.dot(U, np.diag(Sigma))
  # return X_svd


def svd(x):
    u, sigma, vh = np.linalg.svd(x, full_matrices=False, compute_uv=True)
    return u, sigma, vh

def compute_svd_latent(u, sigma, n_comp=None):
    if n_comp is None:
        n_comp = len(sigma)
    latent = np.dot(u[:,:n_comp], np.diag(sigma[:n_comp]))
    return latent


def compute_svd(pattern, n_components):
    u, sigma, vh = svd(pattern)
    latent = compute_svd_latent(u, sigma, n_components)
    results = dict(latent=latent, u=u, sigma=sigma, vh=vh,
                   index=pattern.index)
    return results


def compute_pca(pattern, n_components, return_model=False):
    pca = decomposition.PCA(n_components)
    latent = pca.fit_transform(pattern)
    embeddf = get_embeddf(latent, pattern.index)
    if return_model:
        return embeddf, pca

    return embeddf

def select_and_pca(df, odor_list, window, n_components):
    """
    Select odors and time points and then do pca
    """
    df = select_odors_df(df, odor_list)
    df = select_time_points(df, window)
    embeddf, model = compute_pca(df, n_components, return_model=True)
    return embeddf, model



def compute_umap(pattern, umap_params):
    latent = umap.UMAP(**umap_params).fit_transform(pattern)
    embeddf = get_embeddf(latent, pattern.index)
    return embeddf


def compute_pca_umap(df, n_components, umap_params):
    scaled_df = standard_scale(df)
    pca_latent = compute_pca(scaled_df, n_components)
    umap_latent = compute_umap(pca_latent, umap_params)
    return umap_latent


def get_embeddf(latent, index):
    embeddf = pd.DataFrame(latent, index=index)
    embeddf = embeddf.reindex(index.unique('odor'), level='odor')
    return embeddf

# def plot_embed_2d(embeddf, component_idx, plot_type='line', ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     embeddf = embeddf.iloc[:,list(component_idx)]
#     groups = embeddf.groupby(['odor'])
#     ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
#     for name, group in groups:
#         if plot_type == 'line':
#             ax.plot(group.iloc[:,0], group.iloc[:,1], marker='o',
#                     linestyle='-', ms=4, label=name, alpha=0.7)
#         elif plot_type == 'scatter':
#             ax.scatter(group.iloc[:,0], group.iloc[:,1], marker='o',
#                        s=4, label=name, alpha=0.7)

def plot_embed_1d(embeddf, component_idx, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if "time_index" in embeddf.index.names:
        embeddf.index = embeddf.index.droplevel("time_index")

    embeddf = embeddf.iloc[:,component_idx]
    df = embeddf.unstack().transpose()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    odor_list = embeddf.index.unique('odor').tolist()
    for label, content in df.iteritems():
        color = color_cycle[odor_list.index(label[0])]
        ax.plot(content.to_numpy(), color=color, linestyle='-', ms=4, label=label[0], alpha=0.7)


def get_dimred_df(dimred):
    return pd.DataFrame(dimred['latent'], index=dimred['index'])

def compute_fa(pattern, n_components):
    """
    Factor analysis of neuronal activity
    """
    fa = decomposition.FactorAnalysis(n_components=n_components)
    latent = fa.fit_transform(pattern)
    results = dict(latent=latent, index=pattern.index, fa=fa)
    return results


def _compute_scores(X, ncomp_list):
    pca = decomposition.PCA(svd_solver="full")
    fa = decomposition.FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in ncomp_list:
        print(n)
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))
    return pca_scores, fa_scores


def compute_cv_scores(pattern, max_ncomp, step):
    """
    Compute cross validation scores for choosing best n_components parameters
    for PCA and FA
    """
    ncomp_list = np.arange(0, max_ncomp, step)
    pca_scores, fa_scores = _compute_scores(pattern, ncomp_list)
    results = dict(ncomp_list=ncomp_list, pca_scores=pca_scores,
                   fa_scores=fa_scores)
    return results


def _get_best_ncomp(ncomp_list, scores):
    best_ncomp = ncomp_list[np.argmax(scores)]
    best_score = scores[np.argmax(scores)]
    return best_ncomp, best_score


def get_best_ncomp(results):
    best_results = dict()
    for method in ['pca', 'fa']:
        best_ncomp, best_score = _get_best_ncomp(results['ncomp_list'],
                                                 results[method+'_scores'])
        best_results[method] = dict(best_ncomp=best_ncomp, best_score=best_score)
    return best_results


def plot_on_poincare_disk(embeddf, ax=None, plot_type='scatter'):
    odor_list = embeddf.index.unique('odor').tolist()
    clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    groups = embeddf.groupby(['odor'])
    for name, group in groups:
        cidx = odor_list.index(name)
        color = clr_cycle[cidx]

        trials = group.index.get_level_values('trial').unique()
        for trial in trials:
            trial_data = group.xs(trial, level='trial')
            x = trial_data.iloc[:,0].to_numpy()
            y = trial_data.iloc[:,1].to_numpy()
            z = np.sqrt(1 + np.sum(trial_data**2, axis=1)).to_numpy()
            disk_x = x / (1 + z)
            disk_y = y / (1 + z)

            if plot_type == 'line':
                # ax.quiver(disk_x[:-1], disk_y[:-1],
                #           disk_x[1:]-disk_x[:-1],
                #           disk_y[1:]-disk_y[:-1],
                #           scale_units='xy', angles='xy',
                #           scale=1, color=color)
                cmap = LinearSegmentedColormap.from_list('custom',
                                        [(0, 'white'),
                                         (1, color)], N=256)
                # cmap = 'viridis'
                norm = plt.Normalize(-20, len(disk_x))
                # lwidths = np.arange(len(disk_x)) / 10
                points = np.array([disk_x, disk_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(np.arange(len(disk_x)))
                # linewidths=lwidths,
                ax.add_collection(lc)
                # ax.plot(disk_x, disk_y, label=name, alpha=0.7, color=color)
            ax.scatter(disk_x, disk_y, label=name, marker='+',
                       alpha=0.7, color=color, s=20)

    boundary = plt.Circle((0,0), 1, fc='none', ec='k')
    ax.set_xlim((-1.01, 1.01))
    ax.set_ylim((-1.01, 1.01))
    ax.add_artist(boundary)
    ax.axis('off')


def plot_embed_2d(embeddf, component_idx, ax, plot_type='scatter'):
    odor_list = embeddf.index.unique('odor').tolist()
    clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    embeddf = embeddf.iloc[:,list(component_idx)]
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    groups = embeddf.groupby(['odor'])
    for name, group in groups:
        cidx = odor_list.index(name)
        color = clr_cycle[cidx]

        trials = group.index.get_level_values('trial').unique()
        for trial in trials:
            trial_data = group.xs(trial, level='trial')
            x = trial_data.iloc[:,0]
            y = trial_data.iloc[:,1]

            if plot_type == 'line':
                cmap = LinearSegmentedColormap.from_list('custom',
                                        [(0, 'white'),
                                         (1, color)], N=256)
                norm = plt.Normalize(-20, len(x))
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(np.arange(len(x)))
                ax.add_collection(lc)
            ax.scatter(x, y, label=name, marker='o',
                       alpha=0.7, color=color, s=1)


def plot_latent(embeddf, component_idx, ax=None):
    """
    Plot latent variables as a whole time trace concatenating all trials
    """
    if ax is None:
        fig, ax = plt.subplots()
    df = embeddf.iloc[:,component_idx].to_numpy()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    ax.plot(df)


def plot_latent_3d(latent_df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    groups = latent_df.groupby(['odor'])
    for name, group in groups:
        ax.scatter(group.iloc[:,0], group.iloc[:,1], group.iloc[:,2],
                   marker='o', s=4, label=name, alpha=0.7)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Surface Plot')
