import scipy.io
import pandas as pd
import numpy as np
from os.path import join as pjoin

def compute_area_under_the_curve(params, param_idx, training_day, num_trials_per_day):
    """
    Computes the area under the curve for cs_minus and cs_plus for a given parameter.
    
    params: numpy array of shape (num_cs, num_params, num_trials)
    num_trials_per_day: int, number of trials per day
    param_idx: index of the parameter to compute AUC for
    """
    # Extract data for the given parameter index (for both cs_minus and cs_plus)
    cs_minus_data = params[0, param_idx, :]  # First index (0) corresponds to cs_minus
    cs_plus_data = params[1, param_idx, :]   # Second index (1) corresponds to cs_plus

    num_valid_trials = training_day * num_trials_per_day
    cs_minus_data = cs_minus_data[:num_valid_trials]
    cs_plus_data = cs_plus_data[:num_valid_trials]

    # Compute AUC using np.trapz (or replace with actual logic if different)
    auc_cs_minus = np.trapz(cs_minus_data)
    auc_cs_plus = np.trapz(cs_plus_data)

    return auc_cs_minus, auc_cs_plus


def compute_auc_for_all_parameters(params, training_day, num_trials_per_day):
    """
    Computes the AUC for all parameters for a given fish.
    
    params: numpy array of shape (num_cs, num_params, num_trials)
    num_trials_per_day: int, number of trials per day
    Returns a list of AUCs for each parameter (both cs_minus and cs_plus).
    """
    num_params = params.shape[1]  # Get the number of parameters
    aucs = []
    for param_idx in range(num_params):
        auc = compute_area_under_the_curve(params, param_idx, training_day, num_trials_per_day)
        aucs.append(auc)
    return aucs

def prepare_juvenile_behavior_df():
    behavior_dir = '/tungstenfs/scratch/gfriedri/hubo/behavior/data'
    mat_file = pjoin(behavior_dir, 'juvenile_behavior_params.mat')
    num_trials_per_day = 9

    # Load the .mat file
    mat = scipy.io.loadmat(mat_file)

    # Extract fish_ids, training days, and param names
    fish_ids = [str(fish_id[0][0]) for fish_id in mat['fishIds']]
    training_days = [int(day[0]) for day in mat['days']]
    param_names = [str(param_name[0]) for param_name in mat['paramNames'][0]]  # Get the parameter names
    
    # Handle exceptions in training days
    for idx in [27, 28, 29]:
        training_days[idx] -= 1

    # Extract the behavior data
    all_behavior_data = mat['allData']  # Shape: (num_cs, num_params, num_trials, num_fish)

    aucs_list = []
    
    # Loop over each fish to compute AUCs
    num_fish = all_behavior_data.shape[3]  # Fourth dimension corresponds to fish
    for fish_idx in range(num_fish):
        fish_id = fish_ids[fish_idx]
        params = all_behavior_data[:, :, :, fish_idx]  # Extract params for this fish
        training_day = training_days[fish_idx]
        
        # Compute AUC for all parameters for this fish
        aucs = compute_auc_for_all_parameters(params, training_day, num_trials_per_day)
        aucs_list.append((fish_id, training_day, aucs))

    # Create a DataFrame with fish_id, training_day, and aucs
    behavior_df = pd.DataFrame(aucs_list, columns=['fish_id', 'training_day', 'aucs'])

    # Expand the 'aucs' list into separate columns for each parameter (cs_minus and cs_plus)
    auc_columns = []
    for param_name in param_names:
        auc_columns.extend([
            f'auc_{param_name}_cs_minus',
            f'auc_{param_name}_cs_plus'
        ])

    # Flatten the 'aucs' into individual columns
    aucs_expanded = []
    for aucs in behavior_df['aucs']:
        aucs_row = []
        for auc in aucs:
            aucs_row.extend(auc)
        aucs_expanded.append(aucs_row)

    aucs_df = pd.DataFrame(aucs_expanded, columns=auc_columns)
    behavior_df = pd.concat([behavior_df.drop(columns=['aucs']), aucs_df], axis=1)

    return behavior_df, param_names


def compute_behavior_measures_per_day(behavior_df, param_names):
    """
    Computes the diff_per_day, cs_minus_per_day, and cs_plus_per_day for all parameters in the behavior DataFrame.
    
    behavior_df: DataFrame containing AUC values for each parameter and fish
    param_names: List of parameter names (strings) corresponding to the params in behavior_df
    
    Returns: A DataFrame with the additional computed columns for each parameter.
    """
    behavior_measure_df = behavior_df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    # Loop through each parameter and compute the desired measures
    for param_name in param_names:
        # Compute the difference between cs_plus and cs_minus for the current parameter
        behavior_measure_df[f'auc_{param_name}_diff'] = (
            behavior_measure_df[f'auc_{param_name}_cs_plus'] - behavior_measure_df[f'auc_{param_name}_cs_minus']
        )
        
        # Get the training day to compute the per-day measures
        training_day = behavior_measure_df['training_day']
        
        # Compute diff_per_day
        behavior_measure_df[f'auc_{param_name}_diff_per_day'] = (
            behavior_measure_df[f'auc_{param_name}_diff'] / training_day
        )
        
        # Compute cs_minus_per_day
        behavior_measure_df[f'auc_{param_name}_cs_minus_per_day'] = (
            behavior_measure_df[f'auc_{param_name}_cs_minus'] / training_day
        )
        
        # Compute cs_plus_per_day
        behavior_measure_df[f'auc_{param_name}_cs_plus_per_day'] = (
            behavior_measure_df[f'auc_{param_name}_cs_plus'] / training_day
        )
    
    return behavior_measure_df

