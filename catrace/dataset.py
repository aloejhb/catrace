from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from .utils import load_config


@dataclass_json
@dataclass
class DatasetConfig:
    db_dir: str
    exp_list: list[tuple[str, str]]
    odors: list[str]
    conditions: list[str]
    frame_rate: float
    num_trials: int
    processed_trace_dir: Optional[str] = None
    results_dir: Optional[str] = None
    fig_dir: Optional[str] = None
    odors_stimuli: Optional[list[str]] = None
    odors_aa: Optional[list[str]] = None
    odors_ba: Optional[list[str]] = None
    odors_learned: Optional[list[str]] = None
    odors_novel: Optional[list[str]] = None
    onsets: Optional[list[int]] = None


def load_dataset_config(file_path):
    config = load_config(file_path, DatasetConfig)
    return config


def get_odors_by_key(dsconfig, odor_key):
    # odor_key should be a string starting with 'odors_', if not raise an error
    assert odor_key.startswith('odors_'), 'The key should start with "odors_"'
    # odor_key should be in the dataset config
    assert hasattr(
        dsconfig, odor_key
    ), f'The key "{odor_key}" is not in the dataset config'
    odors = getattr(dsconfig, odor_key)
    return odors
