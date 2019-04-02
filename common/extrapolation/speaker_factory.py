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
    
    # yield Speaker(False, 40, 'speakers_40_clustering_vs_reynolds', dataset="timit")
    # yield Speaker(False, 100, 'speakers_100_50w_50m_not_reynolds', dataset="timit")
    yield Speaker(True, 40, 'speakers_40_clustering_vs_reynolds', dataset="timit")
    # yield Speaker(True, 60, 'speakers_60_clustering', dataset="timit")
    # yield Speaker(True, 80, 'speakers_80_clustering', dataset="timit")
    # yield Speaker(True, 590, 'speakers_590_clustering_without_raynolds', dataset="timit")

    # yield Speaker(True, 40, "vox2_speakers_40", dataset="voxceleb2", max_audio_length=6600)
    # yield Speaker(True, 10, "vox2_speakers_10", dataset="voxceleb2", max_audio_length=6600)
    