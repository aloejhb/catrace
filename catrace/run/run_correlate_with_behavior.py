import pandas as pd
from os.path import join as pjoin
from catrace.dataset import load_dataset_config
from catrace.run.run_distance import read_mats_from_dir, average_over_repeats
from catrace.run.run_distance import get_group_vs_group


def load_distance_per_fish(config_file,
                           distance_dir,
                           measure_name,
                           cs_odors=False,
                           cs_single_direction=0,
                           odor_group1=None,
                           odor_group2=None,
                           deduplicate=False,
                           average_per_fish=False,

):
    dsconfig= load_dataset_config(config_file)

    dist_dir = pjoin(dsconfig.processed_trace_dir, distance_dir)
    exp_list = dsconfig.exp_list
    num_repeats = 50
    all_simdf = read_mats_from_dir(dist_dir, exp_list, num_repeats)
    avg_simdf = average_over_repeats(all_simdf)

    # Remove condition equal to naive
    avg_simdf = avg_simdf[avg_simdf.index.get_level_values('condition') != 'naive']
    if cs_odors:
        # Group avg_simdf by condition
        cond_subsimdfs = []
        for condition, group in avg_simdf.groupby('condition'):
            odors = condition.split('-')
            # capitialize the first letter of each odor
            odors = [odor.capitalize() for odor in odors]

            if cs_single_direction == 0:
                odor_group1 = odors
                odor_group2 = odors
            elif cs_single_direction == 1:
                odor_group1 = [odors[0]]
                odor_group2 = [odors[1]]
            elif cs_single_direction == -1:
                odor_group1 = [odors[1]]
                odor_group2 = [odors[0]]
            else:
                raise ValueError('cs_single_direction should be 0, 1 or -1')
            cond_subsimdf = get_group_vs_group(group,
                                               odor_group1,
                                               odor_group2,
                                               measure_name=measure_name, 
                                               deduplicate=deduplicate)
            cond_subsimdfs.append(cond_subsimdf)
        subsimdf = pd.concat(cond_subsimdfs)
    else:
        subsimdf = get_group_vs_group(avg_simdf, odor_group1, odor_group2, measure_name=measure_name, deduplicate=deduplicate)

    if average_per_fish:
        # Average so that only fish_id and condition is left
        subsimdf_per_fish = subsimdf.groupby(['fish_id', 'condition']).mean()
    else:
        subsimdf_per_fish = subsimdf

    subsimdf_per_fish.reset_index(level='fish_id', inplace=True)
    # repace each string in the fish_id column with '_Dp' to ''
    subsimdf_per_fish['fish_id'] = subsimdf_per_fish['fish_id'].str.replace('_Dp', '')

    subsimdf_per_fish = subsimdf_per_fish.reset_index()
    return subsimdf_per_fish


# def merge_with_behavior(subsimdf_per_fish, behavior_measure_df):
#     merged_behavior_df = pd.concat([subsimdf_per_fish.set_index('fish_id'), behavior_measure_df.set_index('fish_id')], axis=1, join='inner').reset_index()
#     return merged_behavior_df


def merge_with_behavior(subsimdf_per_fish, behavior_measure_df):
    merged_behavior_df = pd.merge(subsimdf_per_fish, behavior_measure_df, on='fish_id', how='inner')
    return merged_behavior_df

# from catrace.stats import plot_regression
# plot_regression(merged_behavior_df, 'auc_zeta_diff_per_day', 'mahal', hue='condition')


from ..dataset import load_dataset_config
from ..stats import plot_regression


def regression_distance_with_behavior(config_file,
                                      metric,
                                      load_distance_per_fish_params,
                                      behavior_measure_df, behavior_measure_name, figsize=(5, 5)):
    dsconfig = load_dataset_config(config_file)

    subsimdf_per_fish = load_distance_per_fish(**load_distance_per_fish_params)
    merged_behavior_df = merge_with_behavior(subsimdf_per_fish, behavior_measure_df)
    fig, model, text_str = plot_regression(merged_behavior_df, metric, behavior_measure_name, hue='condition', figsize=figsize)
    if metric == 'mahal':
        # remove legend
        ax = fig.get_axes()[0]
        ax.legend_.remove()
    return fig, model, text_str