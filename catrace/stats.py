import numpy as np
from scipy.stats import mannwhitneyu


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

def apply_mann_whitney(df):
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
    for odor in df.columns:
        # Split data into naive and trained
        naive_data = df.xs('naive', level='cond')[odor]
        trained_data = df.xs('trained', level='cond')[odor]
        
        # Perform the Mann-Whitney U test
        stat, p = mannwhitneyu(naive_data, trained_data, alternative='two-sided')
        
        # Store results
        results[odor] = {'statistic': stat, 'p-value': p}
    
    return results