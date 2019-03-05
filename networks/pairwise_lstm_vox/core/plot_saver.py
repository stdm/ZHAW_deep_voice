"""
    Provides functionality to store and accurately depict plots.

    Work of Gerber and Glinski.
"""
import matplotlib
import matplotlib.pyplot as plt

from common.utils.paths import *

matplotlib.use('Agg')


def save_accuracy_plot(history, name):
    sav = get_experiment_plots(name + "_acc.png")
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')
    plt.grid()
    plt.savefig(sav)


def save_loss_plot(history, name):
    sav = get_experiment_plots(name + "_loss.png")
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.grid()
    plt.savefig(sav)
