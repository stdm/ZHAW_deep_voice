import json
import os

from common.utils.paths import *

def reset_progress(settings):
    net_dir = get_experiment_nets(settings['SAVE_PATH'])
    if os.path.isfile(net_dir+'/train_progress.csv'):
        os.remove(net_dir+'/train_progress.csv')
    if os.path.isfile(net_dir+'/val_progress.csv'):
        os.remove(net_dir+'/val_progress.csv')

def save_final(net, settings):
    net_dir = get_experiment_nets(settings['SAVE_PATH'])
    if not os.path.isdir(net_dir):
        os.makedirs(net_dir)
    net.save_parameters(net_dir+'/final_epoch')
    with open(net_dir + '/settings.json', 'w') as f:
        json.dump(settings, f)

def save_epoch(net, settings, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=True):
    while len(save_rules) < len(name):
        save_rules.append('n')
    mode = 'train' if train else 'val'

    net_dir = get_experiment_nets(settings['SAVE_PATH'])
    if not os.path.isdir(net_dir):
        os.makedirs(net_dir)
    progress_file = net_dir + '/' + mode + '_progress.csv'

    if not os.path.isfile(progress_file):
        with open(progress_file, 'w+') as f:
            t = 'epoch,time_used,loss'
            for n in name:
                t += ',' + n
            f.write(t+'\n')
    with open(progress_file, 'a') as f:
        t = '%d,%f,%f'%(epoch,time_used,mean_loss)
        for v in indices:
            t += ',%f'%v
        f.write(t+'\n')

    if mode not in best_values:
        vals = {}
        vals['loss'] = mean_loss
        net.save_parameters(net_dir+'/'+mode+'_best_loss')
        for k, v, r in zip(name, indices, save_rules):
            if r == '+' or r == '-':
                vals[k] = v
                net.save_parameters(net_dir+'/'+mode+'_best_'+k)
        best_values[mode] = vals
    else:
        vals = best_values[mode]
        if mean_loss < vals['loss']:
            vals['loss'] = mean_loss
            net.save_parameters(net_dir+'/'+mode+'_best_loss')
        for k, v, r in zip(name, indices, save_rules):
            if (r == '+' and v > vals[k]) or (r == '-' and v < vals[k]):
                vals[k] = v
                net.save_parameters(net_dir+'/'+mode+'_best_'+k)
        best_values[mode] = vals
    return best_values
