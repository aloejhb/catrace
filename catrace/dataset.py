import dataclasses
import dataclasses_json

@dataclasses_json.dataclass_json
@functools.partial(dataclasses.dataclass, frozen=True)
class CatraceDataset:
    """Configuration for calcium traces dataset"""

    exp_list: list(tuple(str, str))
    region_list: list(str)
    frame_rate: 30/4
    exp_info: dict
    odor_list: list(str)
    result_dir: str
    data_root_dir: str
    db_dir: str
    fig_dir: str
    cond_list: list(str)
