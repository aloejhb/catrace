import json
import pandas as pd
from dataclasses_json import DataClassJsonMixin


def copy_index(df1, df2):
    """
    Copy index of df1 to df2 while keep the original df2 index as one level
    """
    df1_index = df1.index.to_frame(index=False)
    df2_index = df2.index.to_frame(index=False)
    dfidx = df2_index.join(df1_index)
    idx = pd.MultiIndex.from_frame(dfidx)
    df2 = df2.set_index(idx)
    return df2


def copy_frame_structure(arr, df):
    """
    Copy the row and column indices and create a new dataframe with values of an array
    """
    assert df.shape == arr.shape, "Dataframe and array shapes should be the same"
    new_df = pd.DataFrame(arr, index=df.index, columns=df.columns)
    return new_df


def load_config(file_path, config_class):
    with open(file_path, 'r') as file:
        data = json.load(file)
        json_str = json.dumps(data)
        config = config_class.from_json(json_str)
        return config

def save_config(config: DataClassJsonMixin, file_path):
    """
    Saves a dataclass_json object to a JSON file.
    
    Args:
    config (DataClassJsonMixin): The dataclass instance to serialize.
    file_path (str): The path to the file where the JSON data should be saved.
    """
    json_str = config.to_json()
    data = json.loads(json_str)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def dict_to_command_line_args(params):
    """
    Converts a dictionary to a command-line argument string.

    Args:
    params (dict): A dictionary where keys are the option names and values are the arguments.

    Returns:
    str: A string formatted as command-line arguments.
    """
    args = []
    for key, value in params.items():
        # Add the key prefixed with '--' and its corresponding value to the args list
        args.append(f"--{key} {value}")
    
    # Join all elements in the list with a space to form the final string
    return ' '.join(args)


def load_config(file_path, config_class):
    with open(file_path, 'r') as file:
        return config_class.from_json(file.read())
    

def load_json_to_dataclass(filename: str, dataclass_type: DataClassJsonMixin):
    """
    Load a JSON file and parse it into a dataclass object.
    
    Args:
    filename (str): The path to the JSON file.
    dataclass_type (Type[T]): The dataclass type to parse the JSON into.
    
    Returns:
    T: An instance of the provided dataclass type.
    
    Raises:
    FileNotFoundError: If the specified file doesn't exist.
    json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(filename, 'r') as file:
            json_data = file.read()
        # Ensure the dataclass is decorated with dataclass_json

        if not hasattr(dataclass_type, 'from_json'):
            raise TypeError(f"{dataclass_type.__name__} must be decorated with @dataclass_json")
        return dataclass_type.from_json(json_data)

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {filename}: {str(e)}", e.doc, e.pos)
