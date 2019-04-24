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

    # @neurudan Speakers for TIMIT?
    # yield Speaker(True, 40, 'timit_speakers_40', dataset="timit")
    # yield Speaker(True, 60, 'timit_speakers_60', dataset="timit")
    # yield Speaker(True, 80, 'timit_speakers_80', dataset="timit")
    # yield Speaker(True, 100, 'timit_speakers_100', dataset="timit")
    # yield Speaker(True, 590, 'timit_speakers_590', dataset="timit")

    # Original TIMIT Speakers
    # yield Speaker(False, 40, 'timit_speakers_40_clustering_vs_reynolds', dataset="timit")
    # yield Speaker(False, 100, 'timit_speakers_100_50w_50m_not_reynolds', dataset="timit")
    # yield Speaker(True, 40, 'timit_speakers_40_clustering_vs_reynolds', dataset="timit")
    # yield Speaker(True, 60, 'timit_speakers_60_clustering', dataset="timit")
    # yield Speaker(True, 80, 'timit_speakers_80_clustering', dataset="timit")
    # yield Speaker(True, 590, 'timit_speakers_590_clustering_without_raynolds', dataset="timit")

    # VoxCeleb2 Speakers
    # lehmacl1@2019-04-16: Since some of the voxceleb2 files are >1min long, setting a high
    # max_audio_length might add huge amounts of "empty" spectrograms with useless information.
    # Trying ~8sec (800) to see if it helps reduce "nonsense audio"
    #
    yield Speaker(False, 10, "vox2_speakers_10_test", dataset="voxceleb2", max_audio_length=800, speakers_per_partition=3)
    # yield Speaker(False, 5994, "vox2_speakers_5994_dev", dataset="voxceleb2", max_audio_length=800, speakers_per_partition=100)
    yield Speaker(False, 120, "vox2_speakers_120_test", dataset="voxceleb2", max_audio_length=800, speakers_per_partition=30)
