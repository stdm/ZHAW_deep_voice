import json
import os

from common.utils.paths import *

def get_params(settings):
    return get_experiment_nets(settings['SAVE_PATH'])+'/final_epoch'

def get_untrained_settings():
    filename = get_experiment_nets('arc_face/all_settings.json')
    settings_tree = {}
    if filename:
        with open(filename, 'r') as f:
            settings_tree = json.load(f)
    save_structure = settings_tree['SAVE_STRUCTURE']
    all_settings = _load_children(settings_tree['DEFAULT'], save_structure)
    untrained_settings = []
    for settings in all_settings:
        print(settings)
        epoch, finished = get_last_epoch(settings)
        print(epoch)
        if not finished:
            untrained_settings.append(settings)
    return untrained_settings

def _dict_equals(x, y):
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    if len(shared_items) == len(x) and len(shared_items) == len(y):
        return True
    return False

def get_trained_settings():
    return _load_trained_children(get_experiment_nets('arc_face'))

def get_last_epoch(settings):
    net_dir = get_experiment_nets(settings['SAVE_PATH'])
    train_path = net_dir+'/train_progress.csv'
    t_l = []
    try:
        with open(train_path, 'r') as f:
            t_l = f.readlines()
        epoch = int(t_l[-1].split(',')[0])
        finished = False
        if epoch >= settings['MAX_EPOCHS'] - 1:
            finished = True
        return epoch, finished
    except:
        return -1, False

def _load_trained_children(path):
    settings = []
    for child in os.listdir(path):
        if child == 'settings.json':
            with open(path+'/'+child, 'r') as f:
                setting = json.load(f)
                epoch, finished = get_last_epoch(setting)
                if finished:
                    settings.append(setting)

        elif os.path.isdir(path+'/'+child):
            settings.extend(_load_trained_children(path+'/'+child))
    return settings

def _load_children(settings, save_structure):
    dicts = []
    if 'CHILDREN' in settings:
        for child in settings['CHILDREN']:
            curr_dict = settings.copy()
            curr_dict.pop('CHILDREN', None)
            for k in child:
                curr_dict[k] = child[k]
            dicts.extend(_load_children(curr_dict, save_structure))
    else:
        curr_dict = settings.copy()
        path = curr_dict['SAVE_PATH']
        for folder_struct in save_structure:
            folder_name = ''
            for k in folder_struct:
                folder_name += k+'='+str(settings[k])+';'
            path += '/'+folder_name[:-1]
        curr_dict['SAVE_PATH'] = path
        dicts.append(curr_dict)
    return dicts
