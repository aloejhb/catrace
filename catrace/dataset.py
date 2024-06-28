from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class DatasetConfig:
    db_dir: str
    exp_list: list[tuple[str, str]]
    odors: list[str]
    conditions: list[str]
    frame_rate: float
    num_trials: int
    fig_dir: str
    odors_stimuli: list[str] = None
    odors_aa: list[str] = None
    odors_ba: list[str] = None
    onsets: list[int] = None