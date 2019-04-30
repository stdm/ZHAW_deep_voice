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
    sav = get_experiment_plots(name + "_acc.png")
    fig = plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')
    plt.grid()
    plt.savefig(sav)


def save_loss_plot(history, name):
    save_loss_plot_direct(name, history.history['loss'], history.history['val_loss'])


def save_loss_plot_direct(name, loss, val_loss):
    sav = get_experiment_plots(name + "_loss.png")
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.grid()
    plt.savefig(sav)
