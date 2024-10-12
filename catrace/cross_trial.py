import pandas as pd
from catrace.visualize import plot_measure, PlotBoxplotParams
from catrace.dataset import get_odors_by_key


def flatten_dataframe(df, col_name):
    df_flattened = pd.DataFrame(df.values.flatten(), columns=[col_name])
    return df_flattened


def flatten_and_concatenate(dfs, df_names, level_name, col_name):
    df_flattend_list = []
    for df in dfs:
        df_flattened = flatten_dataframe(df, col_name)
        df_flattend_list.append(df_flattened)
    concatenated_df = pd.concat(df_flattend_list, keys=df_names, names=[level_name])
    return concatenated_df


def process_cross_trial_similarity(cross_trial_dff, odors, region=None):
    """
    Process cross-trial similarity data by selecting and concatenating specific conditions and regions.

    Parameters:
    - cross_trial_dff (pd.DataFrame): The cross-trial DataFrame with multi-level columns.
    - odors (list): List of odors to filter the columns.

    Returns:
    - pd.DataFrame: The concatenated DataFrame.
    """
    # Select the columns where odor is in odors
    selected_cross_trial_dff = cross_trial_dff.loc[:, cross_trial_dff.columns.get_level_values('odor').isin(odors)]
    
    # Separate naive and trained conditions
    naive_df = selected_cross_trial_dff.xs('naive', level='condition', drop_level=False)
    trained_df = selected_cross_trial_dff.loc[cross_trial_dff.index.get_level_values('condition') != 'naive', :]
    
    # Further separate by region
    if region is not None:
        naive_df = naive_df.xs(region, level='region', drop_level=False)
        trained_df = trained_df.xs(region, level='region', drop_level=False)

    # Flatten and concatenate the DataFrames
    concatenated_df = flatten_and_concatenate([naive_df, trained_df], ['naive', 'trained'], 'condition', 'cosine')
    
    return concatenated_df


# def process_and_plot_cross_trial_similarity(cross_trial_dff, dsconfig, odor_key, params):
#     """
#     Process and plot the cross-trial similarity data by selecting and concatenating specific conditions and regions.

#     Parameters:
#     - cross_trial_dff (pd.DataFrame): The cross-trial DataFrame with multi-level columns.
#     - odors (list): List of odors to filter the columns.

#     Returns:
#     - pd.DataFrame: The concatenated DataFrame.
#     """
#     odors = get_odors_by_key(dsconfig, odor_key)
#     concatenated_df = process_cross_trial_similarity(cross_trial_dff, odors)
#     fig, test_results = plot_measure(concatenated_df, 'cosine', params=params)
#     return fig, test_results

from catrace.visualize import plot_measure_multi_odor_cond, PlotBoxplotMultiOdorCondParams

def process_and_plot_cross_trial_similarity(cross_trial_dff, dsconfig, odor_keys, params: PlotBoxplotMultiOdorCondParams):
    """
    Process and plot the cross-trial similarity data by selecting and concatenating specific conditions and regions.

    Parameters:
    - cross_trial_dff (pd.DataFrame): The cross-trial DataFrame with multi-level columns.
    - odors (list): List of odors to filter the columns.

    Returns:
    - pd.DataFrame: The concatenated DataFrame.
    """
    dfs = []
    for odor_key in odor_keys:
        odors = get_odors_by_key(dsconfig, odor_key)
        concatenated_df = process_cross_trial_similarity(cross_trial_dff, odors)
        dfs.append(concatenated_df)

    multi_df = pd.concat(dfs, keys=odor_keys, names=['odor_key'])
    measure_name = 'cosine'
    fig, ax, test_results = plot_measure_multi_odor_cond(multi_df,
                                                         measure_name,
                                                         odor_name='odor_key',
                                                         condition_name='condition',
                                                         params=params)
    return fig, test_results
