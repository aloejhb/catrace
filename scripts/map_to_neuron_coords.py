# Map cluster ID to neuron coordinates in the anatomy
import sys
import pandas as pd
from skimage.io import imread, imsave
sys.path.append('..')
from catrace.nrn_coord import import_roi_stack, draw_stack, assign_meta

if __name__ == '__main__':
    region = 'OB'
    cluster_df = pd.read_pickle('../../../results/JH_analysis/cluster_df_{}.pkl'.format(region))
    expname = '2021-07-31-DpOBEM-JH17'
    roi_stack_file = '/media/hubo/WD_BoHu/BCE/Ca_imaging/results/2021-07-31-DpOBEM-JH17/OB/roi/roi_stack.tif'
    roi_stack = imread(roi_stack_file)
    rois= import_roi_stack(roi_stack)
    meta_df = cluster_df.loc[cluster_df['fish_id'] == expname]
    rois = assign_meta(rois, meta_df, 'cluster_id')
    clust_stack = draw_stack(rois, roi_stack.shape, 'cluster_id')
    clust_stack_file = '/media/hubo/WD_BoHu/BCE/Ca_imaging/results/2021-07-31-DpOBEM-JH17/OB/analysis/cluster_stack.tif'
    imsave(clust_stack_file, clust_stack)
