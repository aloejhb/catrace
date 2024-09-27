import os
from os.path import join as pjoin
from catrace.exp_collection import process_data_db_parallel
from catrace.process_neuron import (select_cell_type_odors_neurons, save_assembly_results)

@dataclass_json
@dataclass
class RunSelectAssemblyParams:
    pass


def run_select_assembly(params: RunSelectAssemblyParams):
    top_n_per_odor = 50
    select_neuron_window = [37, 60]
    cell_type = None
    in_dir = trace_withid_dir
    method = 'perc'
    threshold = -10
    percentile = 0.05

    select_neuron_tag = f'assembly_window{select_neuron_window[0]}to{select_neuron_window[1]}_{method}' #ptt.get_select_neuron_tag(select_neuron_config)
    if cell_type is not None:
        out_dir = pjoin(in_dir, cell_type, select_neuron_tag)
    else:
        out_dir = pjoin(in_dir, select_neuron_tag)

    overwrite = True
    if not os.path.exists(out_dir) or overwrite:
        os.makedirs(out_dir, exist_ok=True)
        # ptt.save_config(out_dir, select_neuron_config)
        process_data_db_parallel(select_cell_type_odors_neurons, exp_list,
                                out_dir, in_dir,
                                save_func=save_assembly_results,
                                parallelism=16,
                                cell_type=cell_type, odors=dsconfig.odors_stimuli,
                                select_func_name='select_neuron_by_ensemble',
                                window=select_neuron_window,
                                # assembly_size = assembly_size,
                                threshold=threshold,
                                percentile=percentile,
                                method=method)

    select_neuron_dir = out_dir
    print(select_neuron_dir)