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
    times = list(range(12))
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
    # values = np.random.randn(len(row_index), len(columns))

    # Create DataFrame with both multi-indexes
    dff = pd.DataFrame(index=row_index, columns=columns)
    dff = generate_values(dff)
    return dff

def generate_values(dff):
    times = dff.index.get_level_values('time').unique()
    columns = dff.columns
    odors = dff.index.get_level_values('odor').unique()

    # Dictionary to store the template matrices for each odor
    templates = {}

    # Generate template matrices for each odor
    for odor in odors:
        # Initialize the template matrix for current odor
        template = np.zeros((len(times), len(columns)))
        
        for col_idx, column in enumerate(columns):
            plane, neuron, cell_type = column
            # Define step function parameters randomly for each column
            onset = np.random.choice(times)
            amplitude = np.random.uniform(1, 10)

            # Fill the column in the template matrix with the step function
            template[:, col_idx] = [amplitude if time >= onset else 0 for time in times]
        
        templates[odor] = template
    
    # Use the templates to fill the DataFrame, adding noise for each trial
    for (odor, trial, time), _ in dff.iterrows():
        template = templates[odor]
        time_idx = list(times).index(time)
        
        for col_idx, column in enumerate(columns):
            # Base value from the template
            base_value = template[time_idx, col_idx]
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1)  # Small noise unique per trial and time
            
            # Set the value in the DataFrame
            dff.at[(odor, trial, time), column] = base_value + noise

    return dff.astype(float)

if __name__ == '__main__':
    dff = create_test_dataframe()
    print(dff)
