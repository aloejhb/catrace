import pandas as pd
import matplotlib.pyplot as plt
from ..exp_collection import read_df
from ..dataset import get_odors_by_key, DatasetConfig


def plot_avg_trace_with_window(trace_dir, exp_name, window):
    dff = read_df(trace_dir, exp_name)
    avg_trace = dff.groupby(level='time').mean().mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(avg_trace.index.get_level_values('time'), avg_trace.to_numpy())
    ax.axvline(window[0], linestyle='--', color='red')
    ax.axvline(window[1], linestyle='--', color='red')
    return fig, ax


def get_vs_tuple(dsconfig: DatasetConfig, vsname:str):
    group1, group2 = vsname.split('_vs_')
    odor_group1 = get_odors_by_key(dsconfig, f'odors_{group1}')
    odor_group2 = get_odors_by_key(dsconfig, f'odors_{group2}')
    return odor_group1, odor_group2


def get_group_vs_group(dff, odor1_group, odor2_group, measure_name, deduplicate=True):
    # Get tuples of (odor1, odor2) that are in the group
    odor_tuples = [(odor1, odor2) for odor1 in odor1_group for odor2 in odor2_group]
    # Remove the tuples that have the same odor1 and odor2
    odor_tuples = [odor_tuple for odor_tuple in odor_tuples if odor_tuple[0] != odor_tuple[1]]
    
    # Conditionally deduplicate the odor_tuples by sorting and removing duplicates
    if deduplicate:
        odor_tuples = list({tuple(sorted(odor_tuple)) for odor_tuple in odor_tuples})
    
    # Stack the DataFrame so that 'ref_odor' becomes part of the MultiIndex
    dff_stacked = dff.stack()  # This moves the columns (ref_odor) into the index
    
    # Rename the stacked column for clarity
    dff_stacked.name = 'value'
    
    # Use pd.IndexSlice to slice through the MultiIndex
    idx = pd.IndexSlice
    
    # Prepare the filter based on odor_tuples
    filtered_dfs = []
    for odor1, odor2 in odor_tuples:
        # Slice through MultiIndex using `odor1` and `odor2` while preserving other levels
        if 'sample' in dff_stacked.index.names:  # Adjust for additional levels (like 'sample')
            sliced_df = dff_stacked.loc[idx[:, :, :, odor1, odor2]]
        else:  # Default case with two extra levels before 'odor' and 'ref_odor'
            sliced_df = dff_stacked.loc[idx[:, :, odor1, odor2]]
        
        filtered_dfs.append(sliced_df)
    
    # Concatenate all the filtered DataFrames
    gvg = pd.concat(filtered_dfs, keys=odor_tuples)
    
    # Rename the first two levels of the index to 'odor' and 'ref_odor'
    new_index_names = list(gvg.index.names)
    
    if len(new_index_names) >= 2:  # Ensure there are at least two levels to rename
        new_index_names[0] = 'odor'      # First level becomes 'odor'
        new_index_names[1] = 'ref_odor'  # Second level becomes 'ref_odor'
    
    # Apply the new index names
    gvg = gvg.rename_axis(new_index_names, axis=0)
    
    # Convert the final gvg to a DataFrame with the specified column name (measure_name)
    gvg_df = gvg.to_frame(name=measure_name)
    
    return gvg_df