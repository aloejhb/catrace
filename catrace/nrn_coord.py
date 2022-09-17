import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def assign_meta(rois, meta_df, meta_attr):
    for k, roi in enumerate(rois):
        roi_plane = np.ceil((roi.plane + 1) / 2) - 1# 8 plane to original 4 plane * 2 subplanes
        roi_line = meta_df.loc[(meta_df['plane']==roi_plane) &
                        (meta_df['neuron']==roi.tag)]
        if not roi_line.empty:
            val = roi_line[meta_attr].values[0]
            roi.set_meta({meta_attr: val})
        else:
            # skip this roi
            print(f'Skip ROI # {roi.tag} in plane {roi.plane}')
    return rois


def show_stack(stack, figsize=(6, 10), matshow_kwargs=None):
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
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(imgs[-1], cax=cbar_ax)
    return fig
