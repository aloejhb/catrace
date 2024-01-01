import numpy as np

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
