import os

from common.utils.paths import *

def reset_progress(network_file):
    log_dir = get_experiment_logs(network_file)
    if os.path.isfile(log_dir+'/train_progress.csv'):
        os.remove(log_dir+'/train_progress.csv')
    if os.path.isfile(log_dir+'/val_progress.csv'):
        os.remove(log_dir+'/val_progress.csv')

def save_final(net, network_file):
    net_dir = get_experiment_nets(network_file)
    if not os.path.isdir(net_dir):
        os.makedirs(net_dir)
    net.save_parameters(net_dir+'/final_epoch')

def save_epoch(net, network_file, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=True):
    while len(save_rules) < len(name):
        save_rules.append('n')
    mode = 'train' if train else 'val'

    log_dir = get_experiment_logs(network_file)
    net_dir = get_experiment_nets(network_file)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(net_dir):
        os.makedirs(net_dir)
    progress_file = log_dir + '/' + mode + '_progress.csv'

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
