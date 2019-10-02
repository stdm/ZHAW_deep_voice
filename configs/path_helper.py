"""
Single purpose file to determine the absolute path to the "configs" folder.
"""
import inspect
import os.path as path


def get_configs_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return path.dirname(path.abspath(filename))
