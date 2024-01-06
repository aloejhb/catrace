import pandas as pd
from os.path import join as pjoin


def assign_roi_group(df, roi_group_df):
    neurons = df.columns.to_frame(index=False)
    roi_group_df.rename(columns={'roi_tag': 'neuron'}, inplace=True)
    merged_df = pd.merge(neurons, roi_group_df, on=['plane', 'neuron'], how='left')
    multiindex = pd.MultiIndex.from_frame(merged_df[['plane', 'neuron', 'group_tag']])
    df.columns = multiindex
    return df


def read_roi_group_df(neuroi_result_dir, exp_name, region):
    roi_group_df = pd.read_csv(pjoin(neuroi_result_dir, exp_name, region,
                      'roi', 'roi_group_df.csv'))
    return roi_group_df
