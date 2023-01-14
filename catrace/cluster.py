import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator


def plot_clustered_heatmap(all_response, n_clusters, labels, exclude_cluster_id=[]):
    fig = plt.figure(figsize=(20, 30))
    gs = fig.add_gridspec(1, 2,  width_ratios=(9, 2), wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_cid = fig.add_subplot(gs[0, 1], sharey=ax)

    Xcs = get_cluster_df(all_response, labels)

    if len(exclude_cluster_id):
        Xcs = Xcs[~Xcs['cluster_id'].isin(exclude_cluster_id)]

    response_heatmap = ax.matshow(Xcs.iloc[:, :-1], aspect='auto', cmap='viridis')

    ori_cmap = matplotlib.cm.get_cmap('tab20')
    # cnorm = matplotlib.colors.Normalize(vmin=0, vmax=n_clusters-1)
    cnorm = matplotlib.colors.Normalize(vmin=0, vmax=20)
    cmap = matplotlib.colors.ListedColormap([ori_cmap(cnorm(i)) for i in range(n_clusters)])
    # cmap = matplotlib.cm.get_cmap('tab20')

    cluster_id_map = ax_cid.matshow(np.tile(Xcs.cluster_id, (2,1)).transpose(),
                                    aspect='auto', cmap=cmap)
    # plt.colorbar(mappable=cluster_id_map)

    labels, counts = np.unique(Xcs.cluster_id, return_counts=True)
    dummy = [ax_cid.text(0, ct, int(l), fontsize=50) for ct, l in zip(np.cumsum(counts), labels)]

def get_cluster_df(all_response, labels):
    Xc = all_response.append(pd.DataFrame(labels.reshape(1,-1), columns=all_response.columns))
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


def get_cluster_count_df(cluster_df):
    #TODO make this work
    cluster_df = pd.DataFrame(agg_cluster.labels_.reshape(1,-1), columns=all_response.columns)
    cluster_df = cluster_df.transpose().rename(columns={0:'cluster_id'}).reset_index()
    training_dict = dict(dtpar.exp_list)
    cluster_df['train_cond']= cluster_df['fish_id'].map(training_dict)
    cluster_count_df = cluster_df.groupby('train_cond', sort=False).cluster_id.value_counts(normalize=True).sort_index().reindex(dtpar.cond_list, level='train_cond')


def get_cluster_cmap(labels, cmap='tab20c'):
    n_clusters = len(np.unique(labels))
    ori_cmap = matplotlib.cm.get_cmap(cmap)
    cnorm = matplotlib.colors.Normalize(vmin=0, vmax=n_clusters-1)
    cmap = matplotlib.colors.ListedColormap([ori_cmap(cnorm(i)) for i in range(n_clusters)])
    return cmap


def get_cluster_mean_df(H, labels):
    clustdf = get_cluster_df(H, labels)
    cluster_mean_df = clustdf.groupby('cluster_id').mean()
    cluster_mean_df = pd.melt(cluster_mean_df, var_name='trial_key', value_name='response', ignore_index=False).reset_index()

    trial_list = list(cluster_mean_df.trial_key.unique())
    trial_ord = cluster_mean_df.trial_key.map(lambda x: trial_list.index(x))
    cluster_mean_df["trial"] = trial_ord
    cluster_mean_df = cluster_mean_df[cluster_mean_df.trial<18]
    return cluster_mean_df


def _get_odor_list(trial_keys):
    odor_list = [key[0] for key in trial_keys]
    odor_list = list(dict.fromkeys(odor_list))
    return odor_list

def _get_trial_list(trial_keys):
    trial_list = [key[1] for key in trial_keys]
    trial_list = list(dict.fromkeys(trial_list))
    return trial_list


def plot_cluster_tuning(cluster_mean_df, cmap="tab20c"):
    # Initialize the FacetGrid object
    pal = sns.color_palette(cmap) #sns.cubehelix_palette(n_clusters, rot=-.25, light=.7)
    g = sns.FacetGrid(cluster_mean_df, row="cluster_id", hue="cluster_id", aspect=15, height=.5, palette=pal)
    # Draw the densities in a few steps
    g.map(sns.barplot, "trial", 'response')
    # passing color=None to refline() uses the hue mapping
    #g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-.08, .2, int(float(label)), fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=20)

    g.map(label, "trial")
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.05)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    n_trials = cluster_mean_df.trial.max()
    trial_keys = cluster_mean_df.trial_key.unique()
    trial_list = _get_trial_list(trial_keys)
    trial_per_odor = len(trial_list)
    for ax in g.axes.flat:
        ax.set_xticks(np.arange(0, n_trials, trial_per_odor)) # <--- set the ticks first
        ax.set_xticklabels(_get_odor_list(trial_keys), fontsize=20)
        # ax.set_xlabel('Trial')
        ax.set(xlabel=None)
    return g


def plot_cluster_cont_with_stat(cluster_count_df, pairs, cond_list):
    hue_plot_params = dict(x="cluster_id", y="ratio", hue="train_cond", hue_order=cond_list,
                           data=cluster_count_df, palette="Set3")

    with sns.plotting_context("notebook", font_scale = 1.4):
    # Create new plot
        fig, ax = plt.subplots(figsize=(16,10))

    # Plot with seaborn
        ax = sns.boxplot(ax=ax, **hue_plot_params)

    # Add annotations
        annotator = Annotator(ax, pairs, **hue_plot_params)
        annotator.configure(test="t-test_ind").apply_and_annotate()

    # Label and show
        legend = ax.legend()
        frame = legend.get_frame()
        frame.set_facecolor('white')
        ax.set_ylabel('#neurons in cluster/total #neurons')
    return annotator, fig

def get_significant_pairs(annotator):
    new_pairs = [[ano.data.group1, ano.data.group2] for ano in annotator.annotations if ano.data.pvalue < 0.05]
    return new_pairs


def get_embed_df(embedding, labels):
    embed_df = pd.DataFrame(np.column_stack((embedding, labels)), columns=["umap1", "umap2", "cluster_id"])
    embed_df["cluster_id"] = embed_df["cluster_id"].astype(int)
    return embed_df


def plot_embed_df(embed_df, selector=None, cmap="Spectral"):
    fig, ax = plt.subplots(figsize=(10, 10))
    if selector is None:
        data = embed_df
    else:
        data = embed_df[selector]
    if "tuned_odor_family" in embed_df:
        style = "tuned_odor_family"
    else:
        style = None
    g = sns.scatterplot(ax=ax,
                        data=data,
                        x="umap1", y="umap2",
                        hue="cluster_id", style=style,
                        palette=cmap, s=2,
                        legend="full")
    g.set_xlabel("UMAP 1",fontsize=20)
    g.set_ylabel("UMAP 2",fontsize=20)
    g.tick_params(labelsize=14)
    plt.setp(g.get_legend().get_texts(), fontsize='14')
    n_clusters = len(pd.unique(embed_df["cluster_id"]))
    return fig
