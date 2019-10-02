"""
    Loads and checks the config
    Based on previous work of Gygax and Egly.
"""
import configparser


def load_config(path_master_config, path_config):
    config = configparser.ConfigParser()

    # read master config file
    if path_master_config is not None:
        config.read_file(open(path_master_config))

    # read config file
    if path_config is not None:
        config.read_file(open(path_config))

    return config
