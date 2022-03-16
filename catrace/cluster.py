import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def plot_clustered_heatmap(all_response, kmeans, exclude_cluster_id=[]):
    n_clusters = kmeans.n_clusters
    fig = plt.figure(figsize=(20, 30))
    gs = fig.add_gridspec(1, 2,  width_ratios=(9, 2), wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_cid = fig.add_subplot(gs[0, 1], sharey=ax)

    Xcs = get_cluster_df(all_response, kmeans)

    if len(exclude_cluster_id):
        Xcs = Xcs[~Xcs['cluster_id'].isin(exclude_cluster_id)]

    response_heatmap = ax.matshow(Xcs.iloc[:, :-1], aspect='auto')

    ori_cmap = matplotlib.cm.get_cmap('tab20c')
    cnorm = matplotlib.colors.Normalize(vmin=0, vmax=n_clusters-1)
    cmap = matplotlib.colors.ListedColormap([ori_cmap(cnorm(i)) for i in range(n_clusters)])

    cluster_id_map = ax_cid.matshow(np.tile(Xcs.cluster_id, (2,1)).transpose(),
                                    aspect='auto', cmap=cmap)
    # plt.colorbar(mappable=cluster_id_map)

    labels, counts = np.unique(Xcs.cluster_id, return_counts=True)
    dummy = [ax_cid.text(0, ct, int(l), fontsize=50) for ct, l in zip(np.cumsum(counts), labels)]

def get_cluster_df(all_response, kmeans):
    Xc = all_response.append(pd.DataFrame(kmeans.labels_.reshape(1,-1), columns=all_response.columns))
    Xct = Xc.transpose()
    Xct = Xct.set_axis([*Xct.columns[:-1], 'cluster_id'], axis=1, inplace=False) #rename(columns={0:'cluster_id'})
    Xcs = Xct.sort_values('cluster_id')
    return Xcs


def plot_cluster_count(all_response, cluster_labels, exp_list, exclude_cluster_id=[]):
    cluster_df = pd.DataFrame(cluster_labels.reshape(1,-1), columns=all_response.columns)
    cluster_df = cluster_df.transpose().rename(columns={0:'cluster_id'}).reset_index()
    training_dict = dict(exp_list)
    cluster_df['train_cond']= cluster_df['fish_id'].map(training_dict)

    cond_list = ['phe-arg', 'arg-phe', 'phe-trp', 'naive']
    fig, ax = plt.subplots(figsize=(5,15))
    if len(exclude_cluster_id):
        cluster_df = cluster_df[~cluster_df['cluster_id'].isin(exclude_cluster_id)]
    grouped_cluster_id = cluster_df.groupby('train_cond', sort=False).cluster_id
    cluster_count_df = grouped_cluster_id.value_counts(normalize=True).sort_index().reindex(cond_list, level='train_cond')
    cluster_count_df.unstack(0).plot.barh(ax=ax)
    ax.invert_yaxis()
