import yaml, json
import os 

def yaml_convert_config(path):
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