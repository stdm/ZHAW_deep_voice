import sidekit
import os
from common.utils.paths import *


class IvecFeatureExtractor:
    def __init__(self, max_speakers, base_folder, valid_speakers, speaker_list):
        self.max_speakers = max_speakers
        self.base_folder = base_folder
        self.valid_speakers = valid_speakers
        self.speaker_list = speaker_list

    def extract_speaker_data(self):

        speaker_names = []
        file_names = []
        speaker_ids = []
        curr_speaker_num = -1
        old_speaker = ''

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
                               save_param=["vad", "energy", "cep"],
                               keep_all_features=True)

        # Crawl the base and all sub folders
        for root, directories, filenames in os.walk(self.base_folder):

            # Ignore crp and DOC folder
            if root[-5:] not in self.valid_speakers:
                continue

            # Check files
            for filename in filenames:

                # Can't read the other wav files
                if '_RIFF.WAV' not in filename:
                    continue

                # Extract speaker
                speaker = root[-5:]
                if speaker != old_speaker:
                    curr_speaker_num += 1
                    old_speaker = speaker
                    speaker_names.append(speaker)
                    print('Extraction progress: %d/%d' % (curr_speaker_num + 1, self.max_speakers))

                if curr_speaker_num < self.max_speakers: #get_ivec_feature_path(self.speaker_list) + "/feat/"+ speaker + '_' + os.path.splitext(filename)[0]+".h5"
                    fe.save(show=speaker + '_' + os.path.splitext(filename)[0], channel=0, input_audio_filename=root + '/' + os.path.splitext(filename)[0]+'.wav',
                            output_feature_filename=None, noise_file_name=None, snr=10, reverb_file_name=None,
                            reverb_level=-26.0)
                    file_names.append(speaker + '_' + os.path.splitext(filename)[0])
                    speaker_ids.append(str(curr_speaker_num))

        with open(join(get_training('i_vector', self.speaker_list), self.speaker_list +"_files.txt"), "w+") as fh:
            for line in file_names:
                fh.write(line)
                fh.write('\n')

        with open(join(get_training('i_vector', self.speaker_list), self.speaker_list +"_ids.txt"), "w+") as fh:
            for line in speaker_ids:
                fh.write(line)
                fh.write('\n')

