import numpy as np
from scipy.stats import mannwhitneyu, kruskal, ttest_ind
from scikit_posthocs import posthoc_dunn
import pandas as pd


def incremental_histogram(data, bins, chunk_size=10000, normalize=True):
    """
    Compute a histogram from a large 1D array incrementally.

    Parameters:
    - data: The large 1D array.
    - bins: The bin edges for the histogram.
    - chunk_size: The size of data chunks to process at a time.

    Returns:
    - The computed histogram as a NumPy array.
    """
    # Initialize an empty histogram
    histogram = np.zeros(len(bins) - 1, dtype=int)

    # Process the data in chunks
    chunks = list(range(0, len(data), chunk_size))
    chunks.append(len(data))

    for i in range(len(chunks)-1):
        chunk = data[chunks[i]:chunks[i+1]]

        # Update the histogram for the current chunk
        chunk_hist, _ = np.histogram(chunk, bins)

        # Add the chunk's histogram to the overall histogram
        histogram += chunk_hist

    if normalize:
        histogram = histogram / len(data)

    return histogram


def compute_sparseness(x):
    """
    Compute the sparseness of a vector x.

    Args:
    x (numpy.array): A numpy array representing the vector x.

    Returns:
    float: The sparseness value.
    """
    N = len(x)
    if N <= 1:
        return 0  # Sparseness is not well-defined for a single value.
    
    # Compute the sparseness using the given formula
    sum_x = np.sum(x)
    sum_x_squared = np.sum(x**2)
    sparseness = (1 - (sum_x/N)**2 / (sum_x_squared/N)) / (1 - 1/N)
    
    return sparseness

def apply_mann_whitney(df, naive_name='naive', trained_name='trained'):
    """
    Mann Whitney U test for manifold analysis measurements

    Args:
        df (pandas.DataFrame): A pandas DataFrame with the data.
          df.index.names = ['cond', ...]
          df.columns = ['odor1', 'odor2', ...] or [0, 1, ...] representing odor index.
          For the 'cond' level, 'naive' and 'trained' are expected.
    
    Returns:
        results (dict): A dictionary to store the statistical results.
          Keys are odor names and values are dictionaries with keys 'statistic' and 'p-value'.
    """
    results = {}
    for col in df.columns:
        # Split data into naive and trained
        naive_data = df.xs(naive_name, level='cond')[col]
        trained_data = df.xs(trained_name, level='cond')[col]

        # Perform the Mann-Whitney U test
        stat, p = mannwhitneyu(naive_data, trained_data, alternative='two-sided')
        
        # Store results
        results[col] = {'statistic': stat, 'p_value': p}
    
    return results

def apply_test_pair(df, yname=None, group_name1='naive', group_name2='trained', test_type='mannwhitneyu'):
    if 'cond' in df.index.names:
        level = 'cond'
    else:
        level = 'condition'
    
    if yname is None:
        data1 = df.xs(group_name1, level=level)
        data2 = df.xs(group_name2, level=level)
    else:
        data1 = df[df[level] == group_name1][yname]
        data2 = df[df[level] == group_name2][yname]

    if test_type == 'ttest':
        stat, p = ttest_ind(data1, data2, alternative='two-sided')
    elif test_type == 'mannwhitneyu':
        stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
    else:
        raise ValueError("Invalid test_type. Choose either 'ttest' or 'mannwhitneyu'")
    results = {(group_name1, group_name2): {'statistic': stat, 'p_value': p}}
    return results


def apply_test_each_odor_by_cond(df, yname):
    """
    Perform statistical tests for each odor, comparing conditions within each odor.

    Args:
        df: DataFrame with multiindex (fish_id, cond) and each column is an odor_id.
        yname: Name of the column to analyze.

    Returns:
        Dictionary with overall test results and pairwise comparisons against 'naive'.
    """
 
    test_results = {}
    
    # Process each odor separately
    for odor_id in df.columns:
        statdf = df[[odor_id]].reset_index()  # Reset index to access 'cond' and 'fish_id' as regular columns
        statdf.rename(columns={odor_id: yname}, inplace=True)
        # statdf = statdf.reset_index(level='cond')        
        
        # Kruskal-Wallis test
        data_by_condition = [group.values for name, group in statdf.groupby("cond")]
        stat, p_value = kruskal(*data_by_condition)
        
        # Post-hoc Dunn test for pairwise comparisons against 'naive'
        if 'naive' in statdf['cond'].unique():
            dunn_test_results = posthoc_dunn(statdf, val_col=yname, group_col='cond', p_adjust='bonferroni')
            naive_comparisons = dunn_test_results['naive']
        else:
            naive_comparisons = "No naive condition present"
        
        # Store results
        test_results[odor_id] = {
            'Kruskal': {'statistic': stat, 'p_value': p_value},
            'Dunn_naive': naive_comparisons
        }

    return test_results


def apply_test_by_cond(df, yname, naive_name='naive', test_type='kruskal'):
    cond_name = 'condition'
    test_results = {}

    statdf = df.reset_index()
    statdf.rename(columns={df.columns[0]: yname}, inplace=True)
    # Convert categorical condition column to str
    statdf[cond_name] = statdf[cond_name].astype(str)
    # Drop unused columns, to supress warnings
    statdf = statdf[[yname, cond_name]]

    data_by_condition = [group[yname].values for name, group in statdf.groupby("condition", sort=False, observed=True)]
    stat, p_value = kruskal(*data_by_condition)

    if naive_name in statdf[cond_name].unique():
        dunn_test_results = posthoc_dunn(statdf, val_col=yname, group_col=cond_name, p_adjust='bonferroni')
        naive_comparisons = dunn_test_results[naive_name]
    else:
        raise ValueError("No naive condition present")

    test_results = {
        'Kruskal': {'statistic': stat, 'p_value': p_value},
        'Dunn_naive': naive_comparisons
    }

    return test_results


def pool_training_conditions(df, cond_mapping):
    df_pooled = df.copy()

    new_cond = df_pooled.index.get_level_values('condition').map(cond_mapping)

    # Assigne new cond as a new column
    df_pooled['condition'] = new_cond
    # Drop the original 'condition' level from the MultiIndex
    df_pooled = df_pooled.reset_index(level='condition', drop=True)
    # Set the new 'condition' column as the condition level in the MultiIndex
    df_pooled = df_pooled.set_index('condition', append=True)
    df_pooled = sort_conditions(df_pooled, ['naive', 'trained'])
    return df_pooled


def sort_conditions(df, conditions):
    cats = pd.CategoricalDtype(categories=conditions, ordered=True)
    idx = df.index.names.index('condition')
    df.index = df.index.set_levels(df.index.levels[idx].astype(cats),
                                   level='condition')
    # df = df.sort_values('condition') # this alters the order of other levels
    groups = []
    keys = []
    for key, group in df.groupby(level='condition', observed=True):
        groups.append(group)
        keys.append(key)
    df_sorted = pd.concat(groups)
    return df_sorted