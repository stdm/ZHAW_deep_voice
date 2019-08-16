"""
    Provides functionality to store and accurately depict plots.

    Work of Gerber and Glinski.
"""
import matplotlib
import matplotlib.pyplot as plt

from common.utils.paths import *

matplotlib.use('Agg')

def save_accuracy_plot(history, name):
    save_accuracy_plot_direct(name, history.history['acc'], history.history['val_acc'])
    

def save_accuracy_plot_direct(name, acc, val_acc):
    save_plot(name + "_acc", [acc, val_acc], 'epoch', 'accuracy', ['train_acc', 'val_acc'], 'lower right')


def save_loss_plot(history, name):
    save_loss_plot_direct(name, history.history['loss'], history.history['val_loss'])


def save_loss_plot_direct(name, loss, val_loss):
    save_plot(name + "_loss", [loss, val_loss], 'epoch', 'loss', ['train_loss', 'val_loss'], 'upper right')


def save_alr_shape_x_plot(name, shapes_over_time):
    save_plot(name + "_active_learning", shapes_over_time, 'active learning round', 'count', ['X_train', 'X_valid'], 'upper right')


def save_plot(filename, dataparts, xlabel, ylabel, legend, legend_location):
    sav = get_experiment_plots(filename + ".png")
    fig = plt.figure()
    ax = fig.gca()

    for datapart in dataparts:
        plt.plot(datapart)
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc=legend_location)
    plt.grid()
    plt.savefig(sav)
    plt.close()