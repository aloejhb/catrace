import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import decomposition, manifold
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
            ax.plot(
                group.x,
                group.y,
                zs=group.z,
                marker='o',
                linestyle='-',
                ms=4,
                label=name,
                alpha=0.7,
            )
    else:
        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            ax.plot(
                group.x,
                group.y,
                marker='o',
                linestyle='-',
                ms=4,
                label=name,
                alpha=0.7,
            )
    ax.legend()
    return fig


def plot_embed_trial(embeddf):
    odor_list = embeddf['odor'].unique()
    clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for trial in embeddf:
        cidx = odor_list.index(trial.odor)
        ax.plot(
            trial.x,
            trial.y,
            marker='o',
            linestyle='-',
            ms=4,
            label=name,
            alpha=0.7,
            color=clr_cycle[cidx],
        )


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
    points = [
        ax.scatter(
            row['x'],
            row['y'],
            color=color,
            marker='o',
            alpha=index[2] / total_t,
        )
        for index, row in group.iterrows()
    ]
    points[0].label = name
    return points


def svd(x):
    u, sigma, vh = np.linalg.svd(x, full_matrices=False, compute_uv=True)
    return u, sigma, vh


def compute_svd_latent(u, sigma, n_comp=None):
    if n_comp is None:
        n_comp = len(sigma)
    latent = np.dot(u[:, :n_comp], np.diag(sigma[:n_comp]))
    return latent


def compute_svd(pattern, n_components):
    u, sigma, vh = svd(pattern)
    latent = compute_svd_latent(u, sigma, n_components)
    results = dict(latent=latent, u=u, sigma=sigma, vh=vh, index=pattern.index)
    return results


def compute_pca(pattern, n_components=None, return_model=False):
    pca = decomposition.PCA(n_components)
    latent = pca.fit_transform(pattern)
    embeddf = get_embeddf(latent, pattern.index)
    if return_model:
        return embeddf, pca

    return embeddf


def compute_isomap(df, return_model=False, **kwargs):
    model = manifold.Isomap(**kwargs)
    latent = model.fit_transform(df)
    embeddf = get_embeddf(latent, df.index)
    if return_model:
        return embeddf, model

    return embeddf


def compute_embedding(df, func, return_model=False, **kwargs):
    model = func(**kwargs)
    latent = model.fit_transform(df)
    embeddf = get_embeddf(latent, df.index)
    if return_model:
        return embeddf, model

    return embeddf


def select_and_pca(df, odor_list, window, n_components):
    """
    Select odors and time points and then do pca
    """
    df = select_odors_df(df, odor_list)
    df = select_time_points(df, window)
    embeddf, model = compute_pca(df, n_components, return_model=True)
    return embeddf, model


def compute_umap(df, umap_params, scale=False, return_model=False):
    if scale:
        scaled_df = standard_scale(df)
    else:
        scaled_df = df
    umap_model = umap.UMAP(**umap_params)
    latent = umap_model.fit_transform(scaled_df)
    embeddf = get_embeddf(latent, scaled_df.index)
    if return_model:
        return embeddf, umap_model
    return embeddf


def compute_pca_umap(df, n_components, umap_params, scale=False):
    if scale:
        scaled_df = standard_scale(df)
    else:
        scaled_df = df
    pca_latent = compute_pca(scaled_df, n_components)
    umap_latent = compute_umap(pca_latent, umap_params)
    return umap_latent


import numpy as np
import pandas as pd
from sklearn import decomposition

def get_pca_axes_points(pattern):
    """
    Compute a 2D PCA model and return the four points along the first two principal axes:
    center ± PC1, center ± PC2.

    Parameters
    ----------
    pattern : pd.DataFrame
        Input data with shape (n_samples, n_features).
    n_components : int, optional
        Number of PCA components to compute (default=2).

    Returns
    -------
    pca : sklearn.decomposition.PCA
        The fitted PCA model.
    axis_points : pd.DataFrame
        DataFrame containing the four displaced points:
        center+PC1, center-PC1, center+PC2, center-PC2.
    """

    # Fit PCA
    pca = decomposition.PCA(n_components=2)
    pca.fit(pattern)

    # Extract mean (center) and components (directions)
    center = pca.mean_
    pc1_dir = pca.components_[0]
    pc2_dir = pca.components_[1]

    # Amplitudes (standard deviation along each PC)
    pc1_amp = np.sqrt(pca.explained_variance_[0])
    pc2_amp = np.sqrt(pca.explained_variance_[1])

    # Compute the four points in feature space
    center_plus_pc1  = center + pc1_dir * pc1_amp
    center_minus_pc1 = center - pc1_dir * pc1_amp
    center_plus_pc2  = center + pc2_dir * pc2_amp
    center_minus_pc2 = center - pc2_dir * pc2_amp

    # Assemble into DataFrame
    axis_points = pd.DataFrame(
        [center, center_plus_pc1, center_minus_pc1, center_plus_pc2, center_minus_pc2],
        index=["center", "center_plus_pc1", "center_minus_pc1", "center_plus_pc2", "center_minus_pc2"],
        columns=pattern.columns
    )

    return pca, axis_points


def get_embeddf(latent, index):
    embeddf = pd.DataFrame(latent, index=index)
    return embeddf


def plot_embed_1d(embeddf, component_idx, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if 'time_index' in embeddf.index.names:
        embeddf.index = embeddf.index.droplevel('time_index')

    embeddf = embeddf.iloc[:, component_idx]
    df = embeddf.unstack().transpose()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    odor_list = embeddf.index.unique('odor').tolist()
    for label, content in df.iteritems():
        color = color_cycle[odor_list.index(label[0])]
        ax.plot(
            content.to_numpy(),
            color=color,
            linestyle='-',
            ms=4,
            label=label[0],
            alpha=0.7,
        )


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
    pca = decomposition.PCA(svd_solver='full')
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
    results = dict(
        ncomp_list=ncomp_list, pca_scores=pca_scores, fa_scores=fa_scores
    )
    return results


def _get_best_ncomp(ncomp_list, scores):
    best_ncomp = ncomp_list[np.argmax(scores)]
    best_score = scores[np.argmax(scores)]
    return best_ncomp, best_score


def get_best_ncomp(results):
    best_results = dict()
    for method in ['pca', 'fa']:
        best_ncomp, best_score = _get_best_ncomp(
            results['ncomp_list'], results[method + '_scores']
        )
        best_results[method] = dict(
            best_ncomp=best_ncomp, best_score=best_score
        )
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
            x = trial_data.iloc[:, 0].to_numpy()
            y = trial_data.iloc[:, 1].to_numpy()
            z = np.sqrt(1 + np.sum(trial_data**2, axis=1)).to_numpy()
            disk_x = x / (1 + z)
            disk_y = y / (1 + z)

            if plot_type == 'line':
                cmap = LinearSegmentedColormap.from_list(
                    'custom', [(0, 'white'), (1, color)], N=256
                )
                norm = plt.Normalize(-20, len(disk_x))
                points = np.array([disk_x, disk_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(np.arange(len(disk_x)))
                ax.add_collection(lc)
            ax.scatter(
                disk_x,
                disk_y,
                label=name,
                marker='+',
                alpha=0.7,
                color=color,
                s=20,
            )

    boundary = plt.Circle((0, 0), 1, fc='none', ec='k')
    ax.set_xlim((-1.01, 1.01))
    ax.set_ylim((-1.01, 1.01))
    ax.add_artist(boundary)
    ax.axis('off')


def plot_embed_2d(
    embeddf,
    component_idx,
    ax=None,
    manifold_level='manifold', 
    clr_cycle=None,
    markers=None,
    marker_size=2,
    plot_type='scatter',
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    manifold_names = embeddf.index.unique(manifold_level).tolist()
    if clr_cycle is None:
        clr_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    embeddf = embeddf.iloc[:, list(component_idx)]
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    groups = embeddf.groupby(level=manifold_level)
    for name, group in groups:
        cidx = manifold_names.index(name)
        color = clr_cycle[cidx]
        if markers is None:
            marker = 'o'
        else:
            if isinstance(markers, list):
                marker = markers[cidx]
            else:
                raise ValueError("Markers should be a list or a single string.")

        if 'trial' in group.index.names:
            _plot_embed_2d_trial_wise(
                group, ax, name, marker, color, marker_size, plot_type
            )
        else:
            x = group.iloc[:, 0].to_numpy()
            y = group.iloc[:, 1].to_numpy()
            ax.scatter(
                x,
                y,
                label=name,
                marker=marker,
                alpha=0.7,
                color=color,
                s=marker_size,
            )
    return fig, ax


def _plot_embed_2d_trial_wise(group, ax,
                              name, marker, color, marker_size,
                              plot_type='scatter'):
    trials = group.index.get_level_values('trial').unique()
    for trial in trials:
        trial_data = group.xs(trial, level='trial')
        x = trial_data.iloc[:, 0]
        y = trial_data.iloc[:, 1]

        if plot_type == 'line':
            cmap = LinearSegmentedColormap.from_list(
                'custom', [(0, 'white'), (1, color)], N=256
            )
            norm = plt.Normalize(-20, len(x))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.arange(len(x)))
            ax.add_collection(lc)
        ax.scatter(
            x,
            y,
            label=name,
            marker=marker,
            alpha=0.7,
            color=color,
            s=marker_size,
        )


def plot_latent(embeddf, component_idx, ax=None):
    """
    Plot latent variables as a whole time trace concatenating all trials
    """
    if ax is None:
        fig, ax = plt.subplots()
    df = embeddf.iloc[:, component_idx].to_numpy()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    ax.plot(df)


def plot_latent_3d(latent_df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    groups = latent_df.groupby(['odor'])
    for name, group in groups:
        ax.scatter(
            group.iloc[:, 0],
            group.iloc[:, 1],
            group.iloc[:, 2],
            marker='o',
            s=4,
            label=name,
            alpha=0.7,
        )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Surface Plot')
