import torch 
from einops import rearrange, reduce, repeat
import yaml, json
import os 

def yaml_convert_config(path):
    """read yaml file to dict

    Args:
        path (str): yaml file path

    Raises:
        Exception: _description_

    Returns:
        dict: config dict
    """
    if not os.path.exists(path):
        raise Exception("No such file!")
    else:
        with open(path, 'r', encoding='utf-8') as f:
            d = f.read()
        config = yaml.load(d, Loader=yaml.FullLoader)
        return config
    
    
def save_dict_to_yaml(dict_value, save_path):
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

        
def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
    
def str2bool(s:str)->bool:
    return s.lower().strip() == 'true'

def boolean_string(s):
    """_summary_

    Args:
        s (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

