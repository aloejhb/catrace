import os
from os.path import join as pjoin
from catrace.dataset import DatasetConfig
from catrace.utils import load_config
from catrace.exp_collection import process_data_db_parallel
from catrace.process_neuron import (select_cell_type_odors_neurons, save_assembly_results, SelectNeuronParams)


def get_select_neuron_tag(params):
    if params['method'] == 'perc':
        # multiply by 100 to get the percentile and convert to string representing integer using f-string specifying with d
        perc_str = f'{int(params["percentile"]*100):02d}'
        return f'{params["method"]}{perc_str}'
    else:
        return f'{params["method"]}{params["assembly_size"]}'


def run_select_assembly(params: RunSelectAssemblyParams):
    dsconfig= load_config(params.config_file, DatasetConfig)
    exp_list = dsconfig.exp_list
    in_dir = dsconfig.processed_trace_dir

    snparams = params.select_neuron_params
    tag = get_select_neuron_tag(snparams)
    select_neuron_tag = f'assembly_window{snparams.window[0]}to{snparams.window[1]}_{tag}'
    if snparams.cell_type is not None:
        out_dir = pjoin(in_dir, snparams.cell_type, select_neuron_tag)
    else:
        out_dir = pjoin(in_dir, select_neuron_tag)

    if not os.path.exists(out_dir) or params.overwrite:
        os.makedirs(out_dir, exist_ok=True)
        # ptt.save_config(out_dir, select_neuron_config)
        process_data_db_parallel(select_cell_type_odors_neurons, exp_list,
                                 out_dir, in_dir,
                                 save_func=save_assembly_results,
                                 parallelism=params.parallelism,
                                 **snparams.to_dict()
                                 )
    return out_dir