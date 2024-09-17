import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from catrace.mahal import compute_center_euclidean_distance_mat
from catrace.similarity import plot_similarity_mat

# Test function for compute_center_euclidean_distance_mat with 3 odors
def test_compute_center_euclidean_distance_mat():
    # Create a multi-index DataFrame with levels: odor, trial, time
    index = pd.MultiIndex.from_tuples(
        [
            ('A', 1, 1), ('A', 1, 2),
            ('B', 1, 1), ('B', 1, 2),
            ('C', 1, 1), ('C', 1, 2)
        ], 
        names=['odor', 'trial', 'time']
    )
    
    # Create sample neuron data for three odors and two neurons (neurons as columns)
    data = {
        ('neuron1'): [1, 3, 7, 5, 2, 8],  # values for neuron1
        ('neuron2'): [3, 3, 6, 6, 4, 5]   # values for neuron2
    }
    df = pd.DataFrame(data, index=index)

    # Input parameters
    odor_list = ['B', 'C', 'A']  # Changing the order of the odor list
    window = [1, 2]  # Define a window for selecting time points
    
    # Expected result DataFrame (manually calculated based on centers of A, B, and C)
    # Distances are calculated based on the average of neuron1 and neuron2 values for each odor
    expected_dist_mat = pd.DataFrame({
        'B': [0.0, 1.802776, 5.0],   # Distances between B and C, B and A
        'C': [1.802776, 0.0, 3.354102],  # Distances between C and B, C and A
        'A': [5.0, 3.354102, 0.0]    # Distances between A and B, A and C
    }, index=odor_list, columns=odor_list)
    
    # Set index and column names to match the structure
    expected_dist_mat.index.name = 'odor'
    expected_dist_mat.columns.name = 'ref_odor'

    # Call the function to test
    result = compute_center_euclidean_distance_mat(df, odor_list, window)
    # Assert that the result matches the expected output
    assert np.allclose(result.values, expected_dist_mat.values, atol=1e-2), \
        f"Resulting matrix:\n{result}\nExpected matrix:\n{expected_dist_mat}"


# Test plotting the distance matrix
def test_plot_distance_matrix():
    # Create a multi-index DataFrame with levels: odor, trial, time
    index = pd.MultiIndex.from_tuples(
        [
            ('A', 1, 1), ('A', 1, 2),
            ('B', 1, 1), ('B', 1, 2)
        ], 
        names=['odor', 'trial', 'time']
    )
    
    # Create sample neuron data for two neurons (neurons as columns)
    data = {
        ('neuron1'): [1, 3, 7, 5],  # values for neuron1
        ('neuron2'): [3, 3, 6, 6]   # values for neuron2
    }
    df = pd.DataFrame(data, index=index)

    # Input parameters
    odor_list = ['A', 'B']
    window = [1, 2]  # Define a window for selecting time points

    # Call the function to test
    result = compute_center_euclidean_distance_mat(df, odor_list, window)

    fig, ax = plt.subplots()
    # Plot the distance matrix
    plot_similarity_mat(result, ax=ax)