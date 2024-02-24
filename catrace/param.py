import json
from os.path import join as pjoin


def save_config(out_dir, config_class, params):
    """
    Define config and save it into the output dir
    """
    config = config_class(**params)
    config_file = pjoin(out_dir, "config.json")
    with open(config_file, "w") as file:
        json.dump(config.to_json(), file, indent=4)
    return config_file
