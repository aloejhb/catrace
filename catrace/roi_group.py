import json
import pandas as pd
from os.path import join as pjoin


def assign_roi_group(df, roi_group_df, group_names):
    neurons = df.columns.to_frame(index=False)
    roi_group_df.rename(columns={'roi_tag': 'neuron'}, inplace=True)
    mapping_dict = {i+1: name for i, name in enumerate(group_names)}
    roi_group_df['cell_type'] = roi_group_df['group_tag'].map(mapping_dict)
    roi_group_df = roi_group_df.drop('group_tag', axis=1)
    merged_df = pd.merge(neurons, roi_group_df, on=['plane', 'neuron'], how='left')
    multiindex = pd.MultiIndex.from_frame(merged_df[['plane', 'neuron', 'cell_type']])
    df.columns = multiindex
    return df


def read_roi_group_df(neuroi_result_dir, exp_name, region):
    roi_dir = pjoin(neuroi_result_dir, exp_name, region, 'roi')
    group_names_file = pjoin(roi_dir, 'groupNames.json')
    with open(group_names_file, 'r') as gnfile:
        group_names = json.load(gnfile)
    group_str = ''.join(group_names)
    roi_group_df = pd.read_csv(pjoin(roi_dir, f'roi_group_df_{group_str}_corrected.csv'))
    return roi_group_df, group_names
