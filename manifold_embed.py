import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


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
