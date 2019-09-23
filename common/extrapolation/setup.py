"""

"""
from common.extrapolation.speaker_factory import create_all_speakers


def setup_suite(dataset):
    """
    Can be called whenever the project must be setup on a new machine. It automatically
    generates all not yet generated speaker pickles in the right place.
    """
    for speaker in create_all_speakers(dataset):
        if speaker.is_pickle_saved():
            print('{} already exists.'.format(speaker.output_name))
        else:
            speaker.safe_to_pickle()
