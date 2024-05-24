from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class DatasetConfig:
    db_dir: str
    exp_list: list[tuple[str, str]]
    odor_list: list[str]
    cond_list: list[str]
    frame_rate: float
    fig_dir: str