"""
The factory to create all used speaker pickles in the networks.

Based on previous work of Gerber, Lukic and Vogt, adapted by Heusser
"""
from common.extrapolation.speaker import Speaker

# lehmacl1: defines which speakers are being set up initially
#
def create_all_speakers():
    """
    A generator that yields all Speakers that are needed for the Speaker Clustering Suite to function
    :return: yields Speakers
    """

    yield Speaker(False, 40, 'timit_speakers_40', dataset="timit")
    yield Speaker(False, 60, 'timit_speakers_60', dataset="timit")
    yield Speaker(False, 80, 'timit_speakers_80', dataset="timit")
    yield Speaker(False, 100, 'timit_speakers_100', dataset="timit")
    yield Speaker(False, 590, 'timit_speakers_590', dataset="timit")

    yield Speaker(False, 10, "vox2_speakers_10", dataset="voxceleb2", max_audio_length=6600)
    yield Speaker(False, 40, "vox2_speakers_40", dataset="voxceleb2", max_audio_length=6600)
    yield Speaker(False, 100, "vox2_speakers_100", dataset="voxceleb2", max_audio_length=6600)
    yield Speaker(False, 150, "vox2_speakers_150", dataset="voxceleb2", max_audio_length=6600)
