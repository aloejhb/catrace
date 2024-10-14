import pandas as pd
from os.path import join as pjoin
from catrace.dataset import load_dataset_config
from catrace.run.run_distance import read_mats_from_dir, average_over_repeats
from catrace.run.run_distance import get_group_vs_group


def load_distance_per_fish(config_file,
                           distance_dir,
                           odor_group1,
                           odor_group2,
                           measure_name,
                           deduplicate):
    dsconfig= load_dataset_config(config_file)

    dist_dir = pjoin(dsconfig.processed_trace_dir, distance_dir)
    exp_list = dsconfig.exp_list
    num_repeats = 50
    all_simdf = read_mats_from_dir(dist_dir, exp_list, num_repeats)
    avg_simdf = average_over_repeats(all_simdf)

    subsimdf = get_group_vs_group(avg_simdf, odor_group1, odor_group2, measure_name=measure_name, deduplicate=deduplicate)

    # Average so that only fish_id and condition is left
    subsimdf_per_fish = subsimdf.groupby(['fish_id', 'condition']).mean()
    subsimdf_per_fish.reset_index(level='fish_id', inplace=True)
    # repace each string in the fish_id column with '_Dp' to ''
    subsimdf_per_fish['fish_id'] = subsimdf_per_fish['fish_id'].str.replace('_Dp', '')

    subsimdf_per_fish = subsimdf_per_fish.reset_index()
    return subsimdf_per_fish


def merge_with_behavior(subsimdf_per_fish, behavior_measure_df):
    merged_behavior_df = pd.concat([subsimdf_per_fish.set_index('fish_id'), behavior_measure_df.set_index('fish_id')], axis=1, join='inner').reset_index()
    return merged_behavior_df

# from catrace.stats import plot_regression
# plot_regression(merged_behavior_df, 'auc_zeta_diff_per_day', 'mahal', hue='condition')