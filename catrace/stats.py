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

def apply_test_pair(df, yname=None, group_name1='naive', group_name2='trained', test_type='mannwhitneyu', condition_name='condition'):
    import numpy as np
    from scipy.stats import ttest_ind, mannwhitneyu
    from scipy.stats import bootstrap

    level = condition_name
    
    # Extract data for each group
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
    
    # Ensure data1 and data2 are one-dimensional arrays
    data1 = np.asarray(data1).flatten()
    data2 = np.asarray(data2).flatten()

    # Calculate sample sizes
    n1 = len(data1)
    n2 = len(data2)

    # Calculate means and standard deviations
    mean1 = np.mean(data1)
    std1 = np.std(data1, ddof=1)  # Using sample standard deviation (ddof=1)
    mean2 = np.mean(data2)
    std2 = np.std(data2, ddof=1)

    # Perform the statistical test
    if test_type == 'ttest':
        stat, p_value = ttest_ind(data1, data2, alternative='two-sided')
    elif test_type == 'mannwhitneyu':
        stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    elif test_type == 'bootstrap':
        # Combine data for bootstrapping
        data = (data1, data2)

        # Define the statistic to calculate the difference in means
        def _mean_diff(sample1, sample2, axis=-1):
            mean1 = np.mean(sample1, axis=axis)
            mean2 = np.mean(sample2, axis=axis)
            return mean1 - mean2

        # Perform the bootstrap using BCa method for bias-corrected confidence intervals
        res = bootstrap(data, statistic=_mean_diff, n_resamples=10000, method='BCa', paired=False, confidence_level=0.95)

        # Calculate the observed difference
        observed_diff = mean1 - mean2

        # Get the bootstrap distribution
        bootstrap_distribution = res.bootstrap_distribution

        # Calculate the p-value for a two-sided test
        p_value = np.mean(np.abs(bootstrap_distribution) >= np.abs(observed_diff))

        stat = observed_diff
    else:
        raise ValueError("Invalid test_type. Choose 'ttest', 'mannwhitneyu', or 'bootstrap'")

    # Handle cases where stat or p_value are arrays
    if isinstance(stat, np.ndarray):
        stat = stat[0]
    if isinstance(p_value, np.ndarray):
        p_value = p_value[0]

    # Store results including means and stds
    results = {
        (group_name1, group_name2): {
            'statistic': stat,
            'p_value': p_value,
            'n1': n1,
            'n2': n2,
            'mean1': mean1,
            'std1': std1,
            'mean2': mean2,
            'std2': std2
        }
    }
    return results


def format_number(num):
    """
    Formats a number to display in fixed-point notation with enough decimal places to show
    three non-zero digits for small numbers between 0.0001 and 0.01. Uses scientific notation
    for very small or very large numbers.

    Parameters:
    - num (float): The number to format.

    Returns:
    - str: The formatted number as a string.
    """
    if num == 'N/A' or num is None:
        return 'N/A'
    
    num_abs = abs(num)
    if num_abs >= 0.3 and num_abs < 1000:
        # Use fixed-point notation with two decimal places
        return f"{num:.2f}"
    elif num_abs >= 0.0001 and num_abs < 0.3:
        # Calculate the number of decimal places needed to show three significant digits
        decimal_places = int(-math.floor(math.log10(num_abs))) + (3 - 1)
        format_string = f"{{num:.{decimal_places}f}}"
        return format_string.format(num=num)
    else:
        # Use scientific notation for very small or large numbers
        return f"{num:.2e}"


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
        mean1 = result.get('mean1', 'N/A')
        std1 = result.get('std1', 'N/A')
        mean2 = result.get('mean2', 'N/A')
        std2 = result.get('std2', 'N/A')

        # Format the statistics
        statistic_formatted = f"{statistic:.2f}"
        p_value_formatted = format_p_value(p_value)
        mean1_formatted = f"{format_number(mean1)} ± {format_number(std1)}"
        mean2_formatted = f"{format_number(mean2)} ± {format_number(std2)}"

        # Construct the sentence
        sentence = (
            f"Comparing {group_name1} (mean = {mean1_formatted}, n = {n1}) "
            f"vs {group_name2} (mean = {mean2_formatted}, n = {n2}): "
            f"{test_names.get(test_type, 'Statistical test')}, "
            f"{test_stat_symbols.get(test_type, 'stat')} = {statistic_formatted}, {p_value_formatted}."
        )
        return sentence


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

def apply_test_by_cond(df, yname, naive_name='naive', cond_name='condition', test_type='kruskal', return_all_pairs=False):
    from scipy.stats import kruskal
    import pingouin as pg

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
    dunn_results = pg.pairwise_tests(data=statdf, dv=yname, between=cond_name, padjust='bonf')

    # Extract p-values, z-statistics, n-values, and key name
    p_values, z_statistics, n_values, key_name = extract_dunn_statistics(
        dunn_results, statdf, cond_name, naive_name, return_all_pairs)

    # Build test_results
    test_results = {
        'Kruskal': {
            'statistic': stat,
            'p_value': p_value,
            'n': {cond: summary_stats['count'][cond] for cond in summary_stats['count']}
        },
        'mean': summary_stats['mean'],
        'std': summary_stats['std'],
        key_name: {
            'p_values': p_values,
            'z_statistics': z_statistics,
            'n': n_values
        }
    }

    return test_results

def extract_dunn_statistics(dunn_results, statdf, cond_name, naive_name, return_all_pairs):
    p_values = {}
    z_statistics = {}
    n_values = {}

    if not return_all_pairs and naive_name not in statdf[cond_name].unique():
        raise ValueError("No naive condition present")

    for _, row in dunn_results.iterrows():
        group_A = row['A']
        group_B = row['B']
        z_stat = row['T']  # Z statistic
        p_val = row['p-corr']  # Corrected p-value

        # Store sample sizes for both groups
        n1 = statdf[statdf[cond_name] == group_A].shape[0]
        n2 = statdf[statdf[cond_name] == group_B].shape[0]

        # In 'Dunn_naive' mode, only include comparisons involving the naive group
        if not return_all_pairs:
            if group_A == naive_name:
                cond = group_B
                z_stat = -z_stat  # Negate if naive is A
            elif group_B == naive_name:
                cond = group_A
                # z_stat remains the same
            else:
                continue  # Skip comparisons not involving naive
            # Store results with condition as key
            p_values[cond] = p_val
            z_statistics[cond] = z_stat
            n_values[cond] = (n1, n2)
        else:
            # In 'Dunn_all_pairs' mode, store all comparisons
            p_values[(group_A, group_B)] = p_val
            z_statistics[(group_A, group_B)] = z_stat
            n_values[(group_A, group_B)] = (n1, n2)

    key_name = 'Dunn_all_pairs' if return_all_pairs else 'Dunn_naive'
    return p_values, z_statistics, n_values, key_name


def pool_training_conditions(df, cond_mapping, keep_subconditions=False):
    df_pooled = df.copy()

    new_cond = df_pooled.index.get_level_values('condition').map(cond_mapping)

    if keep_subconditions:
        df_pooled['subcondition'] = df_pooled.index.get_level_values('condition')


    # Assigne new cond as a new column
    df_pooled['condition'] = new_cond
    # Drop the original 'condition' level from the MultiIndex
    df_pooled = df_pooled.reset_index(level='condition', drop=True)
    # Set the new 'condition' column as the condition level in the MultiIndex
    df_pooled = df_pooled.set_index('condition', append=True)
    if keep_subconditions:
        df_pooled = df_pooled.set_index('subcondition', append=True)
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
        return f"P = {p_value:.3f}"


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

    # Mean and Std for each condition
    means = test_results['mean']
    stds = test_results['std']

    # Prepare the group statistics sentence
    stats_sentences = []
    for cond in n_per_condition.keys():
        mean = means[cond]
        std = stds[cond]
        n = n_per_condition[cond]
        stats_sentence = f"{cond}: mean = {format_number(mean)} ± {format_number(std)} (n = {n})"
        stats_sentences.append(stats_sentence)

    # Construct the group statistics text
    if len(stats_sentences) == 1:
        stats_text = stats_sentences[0]
    elif len(stats_sentences) == 2:
        stats_text = " and ".join(stats_sentences)
    else:
        stats_text = "; ".join(stats_sentences[:-1])
        stats_text += "; and " + stats_sentences[-1]

    stats_sentence = f"Group statistics: {stats_text}."

    # Detect whether we're dealing with 'Dunn_all_pairs' or 'Dunn_naive'
    if 'Dunn_all_pairs' in test_results:
        return_all_pairs = True
        key_name = 'Dunn_all_pairs'
    elif 'Dunn_naive' in test_results:
        return_all_pairs = False
        key_name = 'Dunn_naive'
    else:
        raise ValueError("Test results do not contain Dunn's test results.")

    dunn_results = test_results[key_name]

    p_values = dunn_results['p_values']
    z_statistics = dunn_results['z_statistics']
    n_values = dunn_results['n']

    # Format comparisons
    comparisons_text = format_comparisons(
        p_values, z_statistics, n_values, n_per_condition, naive_name, return_all_pairs
    )

    if return_all_pairs:
        dunn_sentence = f"Nonparametric multiple comparisons between all groups: {comparisons_text}."
    else:
        n_naive = n_per_condition[naive_name]
        dunn_sentence = f"Nonparametric multiple comparisons against {naive_name} (n = {n_naive}): {comparisons_text}."

    # Combine all sentences
    full_text = f"{kruskal_sentence} {stats_sentence} {dunn_sentence}"

    return full_text

def format_comparisons(p_values, z_statistics, n_values, n_per_condition, naive_name, return_all_pairs):
    comparisons = []

    if return_all_pairs:
        # Handle all pairs comparisons
        for (group_A, group_B) in p_values.keys():
            Q_stat = z_statistics[(group_A, group_B)]
            Q_stat_formatted = f"{Q_stat:.2f}"
            p_value = p_values[(group_A, group_B)]
            p_value_formatted = format_p_value(p_value)
            n1, n2 = n_values[(group_A, group_B)]
            comparison_text = f"{group_A} (n = {n1}) vs {group_B} (n = {n2}), Q = {Q_stat_formatted}, {p_value_formatted}"
            comparisons.append(comparison_text)
    else:
        # Handle comparisons against naive
        n_naive = n_per_condition[naive_name]
        for cond in p_values.keys():
            Q_stat = z_statistics[cond]
            Q_stat_formatted = f"{Q_stat:.2f}"
            p_value = p_values[cond]
            p_value_formatted = format_p_value(p_value)
            n_cond = n_values[cond][1]  # n_values[cond] = (n_naive, n_cond)
            comparison_text = f"{cond}, Q = {Q_stat_formatted}, {p_value_formatted}, n = {n_cond}"
            comparisons.append(comparison_text)

    # Construct the comparisons text
    if len(comparisons) == 1:
        comparisons_text = comparisons[0]
    elif len(comparisons) == 2:
        comparisons_text = " and ".join(comparisons)
    else:
        comparisons_text = "; ".join(comparisons[:-1])
        comparisons_text += "; and " + comparisons[-1]

    return comparisons_text


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
    Formats the test results into a string suitable for a figure caption by calling format_test_results_pair.
    
    Parameters:
    - test_results (dict): Dictionary where keys are odors, and values are dictionaries of test results.
    - test_type (str): The type of test performed ('ttest', 'mannwhitneyu', 'bootstrap').
    
    Returns:
    - str: Formatted string summarizing the test results.
    """
    sentences = []
    for odor in test_results:
        # Get the test result dictionary for this odor
        odor_results = test_results[odor]
        
        # Use format_test_results_pair to format the results for this odor
        sentence = format_test_results_pair(odor_results, test_type=test_type)
        
        # Prepend the odor information
        sentence = f"For {odor}, {sentence}"
        
        sentences.append(sentence)
    
    # Combine all sentences into one string
    full_text = " ".join(sentences)
    return full_text


def format_capacity_test_results_dict(test_results_dict):
    for measure_name, test_results in test_results_dict.items():
        print(measure_name)

        print(format_test_results_pair(test_results['raw']))
        if 'shuffled' in test_results:
            print('shuffled')
            print(format_test_results_pair(test_results['shuffled']))


import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
def plot_regression(dff, x_measure, y_measure, hue=None, ax=None, hue_order=None, figsize=(5, 5)):
    """
    Plots a scatter plot with a regression line on the provided axes, including statistical annotations.

    Parameters:
    - dff: DataFrame containing the data.
    - x_measure: String, name of the column to use for the x-axis.
    - y_measure: String, name of the column to use for the y-axis.
    - hue: String, name of the column to color data points.
    - ax: Axes object on which to draw the plot.
    - hue_order: List or None, specifies the order of categorical hue variable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    # Create scatter plot
    sns.scatterplot(data=dff, x=x_measure, y=y_measure, hue=hue, hue_order=hue_order, ax=ax)

    # Fit linear model using statsmodels to extract R-squared and p-value
    X = sm.add_constant(dff[x_measure])  # Add a constant to the model for the intercept
    model = sm.OLS(dff[y_measure], X).fit()
    predictions = model.predict(X)

    # Get p-value, R-squared, and slope
    p_value = model.f_pvalue
    r_squared = model.rsquared
    slope = model.params[x_measure]

    # Plot the regression line
    ax.plot(dff[x_measure], predictions, color='black', lw=2)

    p_value_srt = format_p_value(p_value)
    print(p_value_srt)
    # Annotate the plot with slope, R², and p-value
    text_str = f'Slope: {slope:.2f}\nR²: {r_squared:.2f}\np-value: {p_value_srt}'
    if slope < 0:
        text_pos = (0.05, 0.25)
    else:
        text_pos = (0.5, 0.25)
    ax.text(text_pos[0], text_pos[1], text_str, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.5),
            fontsize=14)

    # Enhancing the plot
    ax.set_xlabel(x_measure, fontsize=20)
    ax.set_ylabel(y_measure, fontsize=20)
    ax.legend(title=hue, fontsize=14)

    return fig, model, text_str