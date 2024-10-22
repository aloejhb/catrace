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


from ..dataset import load_dataset_config
from ..stats import plot_regression


def regression_distance_with_behavior(config_file,
                                      metric,
                                      distance_dir,
                                      odor_group1,
                                      odor_group2,
                                      behavior_measure_df, behavior_measure_name, figsize=(5, 5)):
    dsconfig = load_dataset_config(config_file)
    measure_name = metric
    deduplicate = True

    subsimdf_per_fish = load_distance_per_fish(config_file, distance_dir, odor_group1, odor_group2, measure_name, deduplicate)

    merged_behavior_df = merge_with_behavior(subsimdf_per_fish, behavior_measure_df)
    fig, model, text_str = plot_regression(merged_behavior_df, behavior_measure_name, metric, hue='condition', figsize=figsize)
    if metric == 'mahal':
        # remove legend
        ax = fig.get_axes()[0]
        ax.legend_.remove()
    return fig, model, text_str