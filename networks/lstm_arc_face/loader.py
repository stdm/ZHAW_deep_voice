import json
import os

from common.utils.paths import *

def get_params(network_file):
    return get_experiment_nets(network_file)+'/val_best_accuracy'

def get_untrained_settings():
    filename = get_experiment_nets('arc_face/all_settings.json')
    settings_tree = {}
    if filename:
        with open(filename, 'r') as f:
            settings_tree = json.load(f)
    all_settings = _load_children(settings_tree['DEFAULT'])
    trained_settings = get_trained_settings()
    untrained_settings = []
    for settings1 in all_settings:
        trained = False
        for settings2 in trained_settings:
            if _dict_equals(settings1, settings2):
                trained = True
        if not trained:
            untrained_settings.append(settings1)
    return untrained_settings

def _dict_equals(x, y):
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    if len(shared_items) == len(x) and len(shared_items) == len(y):
        return True
    return False

def get_trained_settings():
    return _get_children(get_experiment_nets('arc_face'))

def _get_children(path):
    settings = []
    for child in os.listdir(path):
        if child == 'settings.json':
            with open(path+'/'+child, 'r') as f:
                settings.append(json.load(f))
        elif os.path.isdir(path+'/'+child):
            settings.extend(_get_children(path+'/'+child))
    return settings

def _load_children(settings):
    dicts = []
    if 'CHILDREN' in settings:
        for child in settings['CHILDREN']:
            curr_dict = settings.copy()
            settings['CHILDREN'][child]
            curr_dict.pop('CHILDREN', None)
            curr_dict['SAVE_PATH'] += '/' + child
            for k in settings['CHILDREN'][child]:
                curr_dict[k] = settings['CHILDREN'][child][k]
            dicts.extend(_load_children(curr_dict))
    else:
        dicts.append(settings)
    return dicts
