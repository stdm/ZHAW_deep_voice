import sidekit
import os
from common.utils.paths import *


class IvecFeatureExtractor:
    def __init__(self, speaker_list):
        self.speaker_list = speaker_list

    def extract_speaker_data(self, speaker_files):
        curr_speaker_num = 0
        max_speakers = len(speaker_files.keys())
        file_names = []
        speaker_ids = []

        fe = sidekit.FeaturesExtractor(audio_filename_structure="{}.wav",
                               feature_filename_structure="common/data/training/i_vector/"+self.speaker_list+"/feat/{}.h5",
                               sampling_frequency=None,
                               lower_frequency=200,
                               higher_frequency=3800,
                               filter_bank="log",
                               filter_bank_size=24,
                               window_size=0.025,
                               shift=0.01,
                               ceps_number=20,
                               vad="snr",
                               snr=40,
                               pre_emphasis=0.97,
                               save_param=["energy", "cep", "vad"],
                               keep_all_features=True)

        # Crawl the base and all sub folders
        for speaker in speaker_files.keys():
            curr_speaker_num += 1
            speaker_uid = curr_speaker_num

            print('Extraction progress: %d/%d' % (curr_speaker_num, max_speakers))

            # Extract files
            for full_path in speaker_files[speaker]:
                file_name = os.path.splitext(os.path.basename(full_path))[0]
                fe.save(show=speaker + '_' + file_name, channel=0,
                        input_audio_filename=full_path,
                        output_feature_filename=None, noise_file_name=None, snr=10, reverb_file_name=None,
                        reverb_level=-26.0)
                file_names.append(speaker + '_' + file_name)
                speaker_ids.append(str(speaker_uid))

        return file_names, speaker_ids