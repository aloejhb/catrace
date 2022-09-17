import os
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from skimage.io import imread


class Roi:
    def __init__(self, tag, position, plane=None, meta=None):
        self.tag = tag
        self.position = position
        self.plane = plane
        self.meta = meta

    def set_meta(self, meta):
        self.meta = meta


def import_roi_mask(mask):
    rois = []
    tags = np.unique(mask)
    for tag in tags:
        position = np.argwhere(mask == tag)
        roi = Roi(tag, position)
        rois.append(roi)
    return rois


def import_roi_stack(roi_stack):
    rois_list = []
    for plane in range(roi_stack.shape[0]):
        mask = roi_stack[plane, :, :]
        rois = import_roi_mask(mask)
        for roi in rois:
            roi.plane = plane
        rois_list.append(rois)
    all_rois = list(itertools.chain.from_iterable(rois_list))
    return all_rois


def draw_stack(rois, stack_shape, meta_attr):
    stack = np.zeros(stack_shape)
    for roi in rois:
        if roi.meta is not None:
            if meta_attr in roi.meta:
                stack[roi.plane, roi.position[:, 0], roi.position[:, 1]] = \
                  roi.meta[meta_attr]
    return stack


def assign_meta(rois, meta_df, meta_attr, verbal=False):
    for k, roi in enumerate(rois):
        roi_plane = np.ceil((roi.plane + 1) / 2) - 1# 8 plane to original 4 plane * 2 subplanes
        roi_line = meta_df.loc[(meta_df['plane']==roi_plane) &
                        (meta_df['neuron']==roi.tag)]
        if not roi_line.empty:
            val = roi_line[meta_attr].values[0]
            roi.set_meta({meta_attr: val})
        else:
            # skip this roi
            if verbal:
                print(f'Skip ROI # {roi.tag} in plane {roi.plane}')
    return rois


def plot_stack(stack, figsize=(6, 10), matshow_kwargs=None):
    nplane = stack.shape[0]
    ncol = 2
    nrow = np.ceil(nplane/ncol).astype(int)

    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(nrow, ncol)
    gs1.update(wspace=0.05, hspace=0.05)

    axes = []
    imgs = []
    for p in range(nplane):
        ax = plt.subplot(gs1[p])
        img = ax.matshow(stack[p, :, :], aspect='auto', **matshow_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
        imgs.append(img)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
    clb = fig.colorbar(imgs[-1], cax=cbar_ax)
    return fig, clb

# expname, region, data_root_dir,
# meta_df = cluster_df.loc[cluster_df['fish_id'] == expname]
# 'cluster_id'
def map_meta_to_roi_stack(exp_dir, meta_df, meta_attr):
    """
    Map metadata to neuron coordinates in the anatomy stack
    """
    roi_stack_file = os.path.join(exp_dir, 'roi', 'roi_stack.tif')
    roi_stack = imread(roi_stack_file)
    rois= import_roi_stack(roi_stack)
    rois = assign_meta(rois, meta_df, meta_attr)
    # Note that the rois (neurons) not in meta_df are skipped.
    # These are the neurons that are not selected for having large enough respsonse
    mstack = draw_stack(rois, roi_stack.shape, meta_attr)
    return mstack


def plot_meta_stack(mstack, cmap, title=None):
    pal = sns.color_palette(cmap)
    pal.insert(0, (0, 0, 0))
    stack_cmap = matplotlib.colors.ListedColormap(pal)
    matshow_kwargs = dict(vmin=0, vmax=20, cmap=stack_cmap)
    fig, clb = plot_stack(mstack, figsize=(7, 10), matshow_kwargs=matshow_kwargs)
    clb.set_ticks(range(21))
    if title:
        fig.suptitle(title)
    return fig
