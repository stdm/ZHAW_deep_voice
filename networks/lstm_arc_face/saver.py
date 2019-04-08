import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.utils.paths import *

def plot_progress(settings):
    net_dir = get_experiment_nets(settings['SAVE_PATH'])
    train_path = net_dir+'/train_progress.csv'
    val_path = net_dir+'/val_progress.csv'
    t_l, v_l = [], []
    with open(train_path, 'r') as f:
        t_l = f.readlines()
    with open(val_path, 'r') as f:
        v_l = f.readlines()
    title = t_l[0].split(',')
    t_l, v_l = t_l[1:], v_l[1:]
    dic = {'val': {}, 'train': {}}
    for t in title:
        dic['val'][t] = []
        dic['train'][t] = []
    for i in range(len(t_l)):
        if t_l[i].strip() != '':
            t = t_l[i].split(',')
            v = v_l[i].split(',')
            for j in range(len(t)):
                dic['val'][title[j]].append(float(v[j].strip()))
                dic['train'][title[j]].append(float(t[j].strip()))
    x = dic['train'][title[0]]
    for i in range(len(title) - 1):
        t = title[1+i]
        fig = plt.figure(2, clear=True)
        plt.plot(x, dic['train'][t], x, dic['val'][t])
        fig.savefig(net_dir+'/'+t)
        fig.savefig(net_dir+'/'+t+'.svg', format='svg')

def reset_progress(settings):
    net_dir = get_experiment_nets(settings['SAVE_PATH'])
    train_path, val_path = net_dir+'/train_progress.csv', net_dir+'/val_progress.csv'
    t_l, v_l = [], []
    with open(train_path, 'r') as f:
        t_l = f.readlines()
    with open(val_path, 'r') as f:
        v_l = f.readlines()
    if len(t_l) == len(v_l):
        t_l, v_l = t_l[:-1], v_l[:-1]
    else:
        t_l = t_l[:-1]
    os.remove(train_path)
    os.remove(val_path)
    with open(train_path, 'w+') as f:
        f.write(''.join(t_l))
    with open(val_path, 'w+') as f:
        f.write(''.join(v_l))

def save_settings(settings):
    net_dir = get_experiment_nets(settings['SAVE_PATH'])
    try:
        os.makedirs(net_dir)
    except:
        pass
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
    if mode == 'train':
        net.save_parameters(net_dir+'/final_epoch')
    return best_values
