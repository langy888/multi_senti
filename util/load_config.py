import yaml
import sys
sys.path.add("/mnt/lustre/sensebee/backup/fuyubo1/multi_senti/CLMLF")
CONFIG_ROOT = "config/"
def load_config(config_path):
    config_path = CONFIG_ROOT + config_path
    with open(config_path,"r") as f:
        config = yaml.load(f.read(), yaml.FullLoader)
    return config

def merge_config(origin_config, new_dict):
    for k, v in new_dict.items():
        origin_config[k] = v
    return origin_config
