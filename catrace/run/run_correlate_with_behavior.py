import pandas as pd
from os.path import join as pjoin
from catrace.dataset import load_dataset_config
from catrace.run.run_distance import read_mats_from_dir, average_over_repeats
from catrace.run.run_utils import get_group_vs_group
from catrace.run.run_pattern_similarity import read_mats_from_dir as read_sim_mats_from_dir
from catrace.dataset import load_dataset_config


def load_similarity_per_fish(config_file, metric, sub_dir):
    dsconfig = load_dataset_config(config_file)
    sim_dir = pjoin(dsconfig.processed_trace_dir, metric, sub_dir)
    exp_list = dsconfig.exp_list
    # Remove 'naive' from exp_list
    exp_list = [exp for exp in exp_list if exp[1] != 'naive']
    simdf_list, exp_cond_list = read_sim_mats_from_dir(sim_dir, exp_list)

    cond_subsimdfs = []
    for condition, group in zip(exp_cond_list, simdf_list):
        odors = condition.split('-')
        # capitialize the first letter of each odor
        odors = [odor.capitalize() for odor in odors]
        # Average for all odors
        group = group.groupby(level='odor', observed=True).mean().T.groupby(level='odor', observed=True).mean().T
        # Rename the index name to 'odor1'
        group.index.name = 'odor1'
        # Rename the columns name to 'odor2'
        group.columns.name = 'odor2'
        group_stacked = group.stack()

        # Filter odor1 = odors[0] and odor2 = odors[1]
        cond_subsimdf = group_stacked[
            (group_stacked.index.get_level_values('odor1') == odors[0])
            & (group_stacked.index.get_level_values('odor2') == odors[1])
        ].reset_index()
        cond_subsimdfs.append(cond_subsimdf)

    subsimdf_per_fish = pd.concat(cond_subsimdfs, keys=exp_list, names=['fish_id', 'condition'])
    subsimdf_per_fish.reset_index(level='fish_id', inplace=True)
    # repace each string in the fish_id column with '_Dp' to ''
    subsimdf_per_fish['fish_id'] = subsimdf_per_fish['fish_id'].str.replace(
        '_Dp', ''
    )
    subsimdf_per_fish = subsimdf_per_fish.reset_index()
    # Rename column 0 to metric
    subsimdf_per_fish.rename(columns={0: metric}, inplace=True)
    # Remove 'level_1' column
    if 'level_1' in subsimdf_per_fish.columns:
        subsimdf_per_fish.drop(columns=['level_1'], inplace=True)
    return subsimdf_per_fish


def load_distance_per_fish(
    config_file,
    distance_dir,
    measure_name,
    cs_odors=False,
    cs_single_direction=0,
    odor_group1=None,
    odor_group2=None,
    deduplicate=False,
    average_per_fish=False,
    cs_plus_vs_rest=False,
    odors_pool=None,
):

    dsconfig = load_dataset_config(config_file)

    dist_dir = pjoin(dsconfig.processed_trace_dir, distance_dir)
    exp_list = dsconfig.exp_list
    num_repeats = 50
    all_simdf = read_mats_from_dir(dist_dir, exp_list, num_repeats)
    avg_simdf = average_over_repeats(all_simdf)

    # Remove condition equal to naive
    avg_simdf = avg_simdf[
        avg_simdf.index.get_level_values('condition') != 'naive'
    ]
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
            cond_subsimdf = get_group_vs_group(
                group,
                odor_group1,
                odor_group2,
                measure_name=measure_name,
                deduplicate=deduplicate,
            )
            cond_subsimdfs.append(cond_subsimdf)
        subsimdf = pd.concat(cond_subsimdfs)
    elif cs_plus_vs_rest:
        cond_subsimdfs = []
        for condition, group in avg_simdf.groupby('condition'):
            odors = condition.split('-')
            # capitialize the first letter of each odor
            odors = [odor.capitalize() for odor in odors]

            odor_group1 = [odors[0]]  # cs_plus
            # the other odor
            odor_group2 = [
                odor for odor in odors_pool if odor not in odor_group1
            ]
            cond_subsimdf = get_group_vs_group(
                group,
                odor_group1,
                odor_group2,
                measure_name=measure_name,
                deduplicate=deduplicate,
            )
        cond_subsimdfs.append(cond_subsimdf)
        subsimdf = pd.concat(cond_subsimdfs)

    else:
        subsimdf = get_group_vs_group(
            avg_simdf,
            odor_group1,
            odor_group2,
            measure_name=measure_name,
            deduplicate=deduplicate,
        )

    if average_per_fish:
        # Average so that only fish_id and condition is left
        subsimdf_per_fish = subsimdf.groupby(['fish_id', 'condition']).mean()
    else:
        subsimdf_per_fish = subsimdf

    subsimdf_per_fish.reset_index(level='fish_id', inplace=True)
    # repace each string in the fish_id column with '_Dp' to ''
    subsimdf_per_fish['fish_id'] = subsimdf_per_fish['fish_id'].str.replace(
        '_Dp', ''
    )

    subsimdf_per_fish = subsimdf_per_fish.reset_index()
    return subsimdf_per_fish


# def merge_with_behavior(subsimdf_per_fish, behavior_measure_df):
#     merged_behavior_df = pd.concat([subsimdf_per_fish.set_index('fish_id'), behavior_measure_df.set_index('fish_id')], axis=1, join='inner').reset_index()
#     return merged_behavior_df


def merge_with_behavior(subsimdf_per_fish, behavior_measure_df):
    merged_behavior_df = pd.merge(
        subsimdf_per_fish, behavior_measure_df, on='fish_id', how='inner'
    )
    return merged_behavior_df


# from catrace.stats import plot_regression
# plot_regression(merged_behavior_df, 'auc_zeta_diff_per_day', 'mahal', hue='condition')


from ..dataset import load_dataset_config
from ..stats import plot_regression


def regression_distance_with_behavior(
    metric,
    load_distance_per_fish_params,
    behavior_measure_df,
    behavior_measure_name,
    num_baseline_days=None,
    selected_conditions=None,
    plot_regression_params=None,
):
    subsimdf_per_fish = load_distance_per_fish(**load_distance_per_fish_params)
    if selected_conditions is not None:
        subsimdf_per_fish = subsimdf_per_fish[
            subsimdf_per_fish['condition'].isin(selected_conditions)
        ]
    merged_behavior_df = merge_with_behavior(
        subsimdf_per_fish, behavior_measure_df
    )
    # Save behavior_measure_df to a csv file
    behavior_measure_df.to_csv('behavior_measure_df.csv', index=False)
    plot_regression_params = plot_regression_params.to_dict() or {}
    fig, model, text_str = plot_regression(
        merged_behavior_df,
        metric,
        behavior_measure_name,
        hue='condition',
        **plot_regression_params,
    )
    if metric == 'mahal':
        # remove legend
        ax = fig.get_axes()[0]
        ax.legend_.remove()
    if num_baseline_days is not None:
        title = f'Baseline days: {num_baseline_days}'
        fig.suptitle(title)

    fig.tight_layout()
    return fig, model, text_str


def regression_simdf_per_fish_with_behavior(subsimdf_per_fish,
                                            metric,
                                            behavior_measure_df,
                                            behavior_measure_name,
                                            plot_regression_params=None):
    merged_behavior_df = merge_with_behavior(subsimdf_per_fish, behavior_measure_df)
    plot_regression_params = plot_regression_params.to_dict() or {}
    fig, model, text_str = plot_regression(
        merged_behavior_df,
        metric,
        behavior_measure_name,
        hue='condition',
        **plot_regression_params,
    )

    fig.tight_layout()
    return fig, model, text_str
