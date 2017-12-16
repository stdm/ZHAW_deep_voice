import numpy as np
from matplotlib import pyplot as plot

from common.spectogram.spectrogram_converter import mel_spectrogram
from common.utils.paths import *


def save_spectrogramm_png(path):
    # Load the mel spectrogram
    spectrogram = mel_spectrogram(path)

    # Begin the plot
    figure = plot.figure(1)
    plot.imshow(spectrogram[:, 20:160])

    # Add the color bar
    color_bar = plot.colorbar()
    n = np.linspace(0, 35, num=11)
    labels = []
    for l in n:
        labels.append(str(l) + ' dB')
    color_bar.ax.set_yticklabels(labels)

    # Add x and y labels
    plot.xlabel('Spektra (in Zeit)')
    plot.ylabel('Frequenz-Datenpunkte')

    # Save the figure to disc
    figure.savefig(get_result_png('spectrogram'))


if __name__ == '__main__':
    save_spectrogramm_png('SA1_RIFF.wav')
