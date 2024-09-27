import pandas as pd
from os.path import join as pjoin
from .process_time_trace import select_time_points, select_odors_and_sort
from .exp_collection import update_df


def append_neuron_id(dff):
    neuron_id = pd.RangeIndex(start=0, stop=dff.shape[1], step=1)
    if 'plane' in dff.columns.names:
        dff.columns = pd.MultiIndex.from_arrays([dff.columns.get_level_values('plane'),
                                            dff.columns.get_level_values('neuron'),
                                            neuron_id],
                                            names=['plane', 'neuron', 'neuron_id'])
    elif 'neuron' in dff.columns.names:
        dff.columns = pd.MultiIndex.from_arrays([dff.columns.get_level_values('neuron'),
                                            neuron_id],
                                            names=['neuron', 'neuron_id'])
    else:
        # Append a new level called neuron_id to the columns
        dff.columns = pd.MultiIndex.from_arrays([dff.columns, neuron_id], names=[dff.columns.name, 'neuron_id'])
    return dff


def find_top_indices(row, assembly_size):
    # Return only the 'neuron_id' values for the top indices
    return row.nlargest(assembly_size).index.get_level_values('neuron_id')

def find_top_indices_perc(row, threshold, percentile):
    # Filter the row to include only neurons with response greater than the threshold
    filtered_row = row[row > threshold]
    # Calculate the number of top neurons to select based on the 32nd percentile
    top_count = int(len(filtered_row) * percentile)
    # Return only the 'neuron_id' values for the top indices of the filtered row
    return filtered_row.nlargest(top_count).index.get_level_values('neuron_id')

def get_assembly_positions(df, method='top_indices', **kwargs):
    assemblies = {}
    if method == 'top_indices':
        find_func = find_top_indices
        args = dict(assembly_size=kwargs['assembly_size'])
    elif method == 'perc':
        find_func = find_top_indices_perc
        args = dict(threshold=kwargs['threshold'], percentile=kwargs['percentile'])

    for odor, row in df.iterrows():
        indices = find_func(row, **args)
        assemblies[odor] = indices
    return assemblies

def select_neuron_by_assembly(dff, window, method='top_indices', **kwargs):
    dff_in_window = select_time_points(dff, window)
    response = dff_in_window.groupby(level='odor', sort=False).mean()
    
    # Assembly dictionary mapping odors to neuron_ids
    assembly_dict = get_assembly_positions(response, method=method, **kwargs)
    
    # Initialize a DataFrame for assembly membership
    all_neuron_ids = sorted(set().union(*assembly_dict.values()))
    assembly_matrix = pd.DataFrame(False, index=all_neuron_ids, columns=assembly_dict.keys(), dtype=bool)
    
    # Populate the DataFrame with True for neuron_ids that are part of each odor's assembly
    for odor, neuron_ids in assembly_dict.items():
        assembly_matrix.loc[neuron_ids, odor] = True
    
    # Select dff data for neuron_ids that are part of any assembly
    dff_select = dff.loc[:, dff.columns.get_level_values('neuron_id').isin(all_neuron_ids)]
    
    results = {'dff_select': dff_select, 'assembly_matrix': assembly_matrix}
    return results


def save_assembly_results(results, out_dir, exp_name):
    update_df(results['dff_select'], out_dir, exp_name)
    filename = pjoin(out_dir, f'{exp_name}_assembly_matrix.csv')
    results['assembly_matrix'].to_csv(filename)


def select_cell_type_odors_neurons(dff, cell_type, odors, select_func_name, **kwargs):
    if cell_type is not None:
        dff = dff.xs(cell_type, level='cell_type', axis=1)
    dff = select_odors_and_sort(dff, odors)
    if select_func_name == 'select_neuron_by_assembly':
        dff = select_neuron_by_assembly(dff, **kwargs)
    else:
        raise ValueError(f'Unrecognized select_func_name {select_func_name}. So far only select_neuron_by_assembly is supported.')
    return dff


from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Tuple, Optional


@dataclass_json
@dataclass
class SelectNeuronParams:
    cell_type: str
    odors: List[str]
    window: Tuple[float, float]
    method: str
    select_func_name: str = 'select_neuron_by_assembly'
    assembly_size: Optional[int] = None
    threshold: Optional[float] = None
    percentile: Optional[float] = None

    def __post_init__(self):
        if self.method == 'top_indices':
            if self.assembly_size is None:
                raise ValueError("assembly_size must be provided when method is 'top_indices'.")
            if self.threshold is not None or self.percentile is not None:
                raise ValueError("threshold and percentile must be None when method is 'top_indices'.")
        elif self.method == 'perc':
            if self.assembly_size is not None:
                raise ValueError("assembly_size must be None when method is 'perc'.")
            if self.threshold is None or self.percentile is None:
                raise ValueError("threshold and percentile must be provided when method is 'perc'.")
        else:
            raise ValueError(f"Invalid method '{self.method}'. Supported methods are 'top_indices' and 'perc'.")
