import numpy as np
import math

from scipy.stats import mannwhitneyu, kruskal, ttest_ind, bootstrap, norm
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
    if 'cond' in df.index.names or 'cond' in df.columns.names:
        level = 'cond'
    else:
        level = 'condition'
    
    
    if yname is None:
        data1 = df.xs(group_name1, level=level)
        data2 = df.xs(group_name2, level=level)
    else:
        if level in df.index.names:
            data1 = df[df.index.get_level_values(level) == group_name1][yname]
            data2 = df[df.index.get_level_values(level) == group_name2][yname]
        elif level in df.columns.names:
            data1 = df[df[level] == group_name1][yname]
            data2 = df[df[level] == group_name2][yname]
        else:
            raise ValueError("Level not found in index or columns")
    # Calculate sample sizes
    n1 = len(data1)
    n2 = len(data2)

    if test_type == 'ttest':
        stat, p_value = ttest_ind(data1, data2, alternative='two-sided')
    elif test_type == 'mannwhitneyu':
        stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    elif test_type == 'bootstrap':
        # Combine data for bootstrapping
        data = (data1.to_numpy(), data2.to_numpy())
        # Define the statistic to calculate the difference in means
        def _mean_diff(sample1, sample2, axis=-1):
            mean1 = np.mean(sample1, axis=axis)
            mean2 = np.mean(sample2, axis=axis)
            return mean1 - mean2

        # Perform the bootstrap using BCa method for bias-corrected confidence intervals
        res = bootstrap(data, statistic=_mean_diff, n_resamples=10000, method='BCa', paired=False, confidence_level=0.95)
        # Calculate the observed difference
        observed_diff = _mean_diff(data[0].T, data[1].T)
        # Get the bootstrap distribution
        bootstrap_distribution = res.bootstrap_distribution
        print(np.mean(np.abs(bootstrap_distribution)))
        print(np.std(np.abs(bootstrap_distribution)))
        print(np.abs(observed_diff))
        # Calculate the p-value for a two-sided test
        p_value = np.mean(np.abs(bootstrap_distribution) >= np.abs(observed_diff))
        print(p_value)
        stat = observed_diff
    else:
        raise ValueError("Invalid test_type. Choose either 'ttest' or 'mannwhitneyu'")
    
    if isinstance(stat, np.ndarray):
        stat = stat[0]
    if isinstance(p_value, np.ndarray):
        p_value = p_value[0]
    results = {(group_name1, group_name2): {'statistic': stat,
                                            'p_value': p_value,
                                            'n1': n1,
                                            'n2': n2}}
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


import pingouin as pg

def apply_test_by_cond(df, yname, naive_name='naive', test_type='kruskal'):
    cond_name = 'condition'
    test_results = {}

    # If dataframe has a multiindex, reset it
    statdf = df.reset_index()
    statdf.rename(columns={df.columns[0]: yname}, inplace=True)
    # Convert categorical condition column to str
    statdf[cond_name] = statdf[cond_name].astype(str)
    # Drop unused columns, to suppress warnings
    statdf = statdf[[yname, cond_name]]

    # Prepare data by condition
    data_by_condition = [group[yname].values for name, group in statdf.groupby(cond_name, sort=False, observed=True)]
    
    # Perform Kruskal-Wallis test
    stat, p_value = kruskal(*data_by_condition)

    # Calculate mean, std, and n for each condition
    summary_stats = statdf.groupby(cond_name)[yname].agg(['mean', 'std', 'count']).to_dict()

    # Perform Dunn's posthoc test using pingouin
    if naive_name in statdf[cond_name].unique():
        # Perform Dunn's test and get the results
        dunn_results = pg.pairwise_tests(data=statdf, dv=yname, between=cond_name, padjust='bonf')

        # Filter results to only include comparisons involving the naive group
        naive_comparisons = dunn_results[(dunn_results['A'] == naive_name) | (dunn_results['B'] == naive_name)]

        # Create a dictionary to store p-values and Z statistics for comparisons with naive
        p_values = {}
        z_statistics = {}
        n_values = {}
        for _, row in naive_comparisons.iterrows():
            # Identify the other group being compared to naive
            if row['A'] == naive_name:
                cond = row['B']
                # If naive is in A, negate the Z statistic
                z_statistics[cond] = -row['T']
            else:
                cond = row['A']
                # If naive is in B, use the Z statistic as is
                z_statistics[cond] = row['T']
                
            # Store the p-value
            p_values[cond] = row['p-corr']
            # Store the sample size for the comparison
            n_values[cond] = (statdf[statdf[cond_name] == naive_name].shape[0], statdf[statdf[cond_name] == cond].shape[0])

        # Store results in the same format as your original structure
        test_results = {
            'Kruskal': {'statistic': stat, 'p_value': p_value, 'n': {cond: summary_stats['count'][cond] for cond in summary_stats['count']}},
            'mean': summary_stats['mean'],
            'std': summary_stats['std'],
            'Dunn_naive': {
                'p_values': p_values,
                'z_statistics': z_statistics,
                'n': n_values
            }
        }
    else:
        raise ValueError("No naive condition present")

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


def format_p_value(p_value):
    # Set a minimum threshold for p_value to avoid math domain errors
    min_p_value = 1e-300  # Adjust this value as needed for your context
    
    if p_value <= 0 or p_value < min_p_value:
        # Handle p_values that are zero or extremely small
        exponent = int(math.floor(math.log10(min_p_value)))
        return f"P < 1 × 10^{exponent}"
    elif p_value < 1e-4:
        exponent = int(math.floor(math.log10(p_value)))
        base = p_value / (10 ** exponent)
        # Ensure base is between 1 and 10
        return f"P = {base:.1f} × 10^{exponent}"
    elif p_value < 0.001:
        return f"P = {p_value:.4f}"
    else:
        return f"P = {p_value:.2f}"


def format_test_results_by_cond(test_results, naive_name='naive'):
    # Kruskal–Wallis test results
    n_per_condition = test_results['Kruskal']['n']  # dict of n per condition
    total_n = sum(n_per_condition.values())
    degrees_of_freedom = len(n_per_condition) - 1
    H = test_results['Kruskal']['statistic']
    P_value = test_results['Kruskal']['p_value']

    # Format Kruskal–Wallis test results
    H_formatted = f"{H:.2f}"
    P_value_formatted = format_p_value(P_value)
    kruskal_sentence = f"(Kruskal–Wallis test, n = {total_n}, d.f. = {degrees_of_freedom}, H = {H_formatted}, {P_value_formatted})."
    
    # Dunn's test results
    n_naive = n_per_condition[naive_name]
    comparisons = []
    p_values = test_results['Dunn_naive']['p_values']
    z_statistics = test_results['Dunn_naive']['z_statistics']
    n_values = test_results['Dunn_naive']['n']
    
    # Get condition names excluding naive
    condition_names = list(p_values.keys())
    
    for cond in condition_names:
        Q_stat = z_statistics[cond]
        Q_stat_formatted = f"{Q_stat:.2f}"
        p_value = p_values[cond]
        p_value_formatted = format_p_value(p_value)
        n_cond = n_values[cond][1]  # n_values[cond] = (n_naive, n_cond)
        comparison_text = f"{cond}, Q = {Q_stat_formatted}, {p_value_formatted}, n = {n_cond}"
        comparisons.append(comparison_text)
    
    # Now, construct the comparisons text
    if len(comparisons) == 1:
        comparisons_text = comparisons[0]
    elif len(comparisons) == 2:
        comparisons_text = " and ".join(comparisons)
    else:
        comparisons_text = "; ".join(comparisons[:-1])
        comparisons_text += "; and " + comparisons[-1]
    
    # Now, construct the Dunn's test sentence
    dunn_sentence = f"Nonparametric multiple comparisons against {naive_name} (n = {n_naive}): {comparisons_text}."

    # Combine all sentences
    full_text = f"{kruskal_sentence} {dunn_sentence}"
    
    return full_text


def apply_tests_multi_odor_two_cond(sub_mean_madff, yname, odor_name, test_type='mannwhitneyu'):
    """
    Apply tests to subgroups of a DataFrame grouped by odor.

    Parameters:
    - sub_mean_madff (pd.DataFrame): The DataFrame containing the data.
    - odor_name (str): The column name to group by.
    - test_type (str): The type of test to apply. Default is 'kruskal'.

    Returns:
    - dict: A dictionary with odors as keys and test results as values.
    """
    test_results = {}
    for odor, subdf in sub_mean_madff.groupby(odor_name):
        test_results[odor] = apply_test_pair(subdf, yname=yname, test_type=test_type)
    return test_results


def format_test_results_multi_odor_two_cond(test_results, test_type='mannwhitneyu'):
    """
    Formats the test results into a string suitable for a figure caption.
    
    Parameters:
    - test_results (dict): Dictionary where keys are odors, and values are dictionaries of test results.
    - test_type (str): The type of test performed ('ttest', 'mannwhitneyu', 'bootstrap').
    
    Returns:
    - str: Formatted string summarizing the test results.
    """
    test_names = {
        'ttest': 't-test',
        'mannwhitneyu': 'Mann–Whitney U test',
        'bootstrap': 'Bootstrap test'
    }
    test_stat_symbols = {
        'ttest': 't',
        'mannwhitneyu': 'U',
        'bootstrap': 'difference'
    }
    
    sentences = []
    for odor in test_results:
        # Get the test result dictionary for this odor
        odor_results = test_results[odor]
        # odor_results is a dictionary where key is (group_name1, group_name2), value is {'statistic': stat, 'p_value': p_value, 'n1': n1, 'n2': n2}
        # Assuming only one key in odor_results
        for group_pair, result in odor_results.items():
            group_name1, group_name2 = group_pair
            statistic = result['statistic']
            p_value = result['p_value']
            n1 = result['n1']
            n2 = result['n2']
            statistic_formatted = f"{statistic:.2f}"
            p_value_formatted = format_p_value(p_value)
            # Now, construct the sentence
            sentence = (f"For {odor}, comparing {group_name1} (n={n1}) vs {group_name2} (n={n2}): "
                        f"{test_names[test_type]}, {test_stat_symbols[test_type]} = {statistic_formatted}, {p_value_formatted}.")
            sentences.append(sentence)
    
    # Combine all sentences into one string
    full_text = " ".join(sentences)
    return full_text


def format_test_results_pair(test_results, test_type='mannwhitneyu'):
    """
    Formats the test results from apply_test_pair into a string suitable for a figure caption.

    Parameters:
    - test_results (dict): Dictionary where keys are (group_name1, group_name2), and values are dictionaries of test results.
    - test_type (str): The type of test performed ('ttest', 'mannwhitneyu', 'bootstrap').

    Returns:
    - str: Formatted string summarizing the test results.
    """
    test_names = {
        'ttest': 't-test',
        'mannwhitneyu': 'Mann–Whitney U test',
        'bootstrap': 'Bootstrap test'
    }
    test_stat_symbols = {
        'ttest': 't',
        'mannwhitneyu': 'U',
        'bootstrap': 'Δ'
    }
    
    # Since test_results should have only one key-value pair
    if len(test_results) != 1:
        raise ValueError("test_results should contain exactly one comparison result.")
    
    # Extract the comparison result
    for group_pair, result in test_results.items():
        group_name1, group_name2 = group_pair
        statistic = result['statistic']
        p_value = result['p_value']
        n1 = result.get('n1', 'N/A')
        n2 = result.get('n2', 'N/A')
        statistic_formatted = f"{statistic:.2f}"
        p_value_formatted = format_p_value(p_value)
        # Construct the sentence
        sentence = (f"Comparing {group_name1} (n={n1}) vs {group_name2} (n={n2}): "
                    f"{test_names.get(test_type, 'Statistical test')}, "
                    f"{test_stat_symbols.get(test_type, 'stat')} = {statistic_formatted}, {p_value_formatted}.")
        return sentence
