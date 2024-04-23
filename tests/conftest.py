import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def create_test_dataframe():
    # Row multi-index setup
    odors = ['apple', 'banana', 'cherry', 'date']
    trials = [0, 1, 2]
    times = list(range(6))
    row_index = pd.MultiIndex.from_product([odors, trials, times], names=['odor', 'trial', 'time'])

    # Column multi-index setup with specific cell types per neuron and plane
    planes = [0, 1]
    neurons = {0: [0, 1, 2], 1: [0, 1]}  # Neuron indices per plane
    cell_types = {0: ['ct1', 'ct1', 'ct2'], 1: ['ct2', 'ct1']}  # Mapping neuron index to cell_type

    # Create the product of the indices for the columns
    columns = pd.MultiIndex.from_tuples(
        [(plane, neuron, cell_type)
         for plane in planes
         for neuron, cell_type in zip(neurons[plane], cell_types[plane])],
        names=['plane', 'neuron', 'cell_type']
    )

    # Generate random values for the DataFrame
    data_size = len(row_index) * len(columns)  # Total number of elements in the DataFrame
    values = np.random.randint(1, 10, size=data_size).reshape(len(row_index), len(columns))

    # Create DataFrame with both multi-indexes
    dff = pd.DataFrame(values, index=row_index, columns=columns)
    return dff

if __name__ == '__main__':
    dff = create_test_dataframe()
    print(dff)
