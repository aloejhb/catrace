import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import cross_val_score

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


def compute_pca(pattern, n_components):
    pca = decomposition.PCA(n_components)
    latent = pca.fit_transform(pattern)
    results = dict(latent=latent, index=pattern.index, pca=pca)
    return results


def plot_embed_2d(results, component_idx, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    embeddf = pd.DataFrame(results['latent'][:,component_idx],
                           columns=['x', 'y'],
                           index=results['index'])
    groups = embeddf.groupby(['odor'])
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='-', ms=4, label=name, alpha=0.7)


def plot_embed_1d(results, component_idx, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    embeddf = pd.DataFrame(results['latent'][:,component_idx],
                           columns=['x'],
                           index=results['index'])
    df = embeddf.unstack().transpose()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    odor_list = list(df.columns.levels[0])
    for label, content in df.iteritems():
        color = color_cycle[odor_list.index(label[0])]
        ax.plot(content.to_numpy(), color=color, linestyle='-', ms=4, label=label[0], alpha=0.7)

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


if __name__ == '__main__':
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    n_samples, n_features = X.shape
    n_neighbors = 30


    # Plot images of the digits
    n_img_per_row = 20
    img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 10 * i + 1
        for j in range(n_img_per_row):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('A selection from the 64-dimensional digits dataset')
