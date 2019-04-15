import os
import pydub

source = input('enter the source, please:     ')
destination = input('enter the destination please: ')

source += '/' if source[-1] != '/' else ''
destination += '/' if destination[-1] != '/' else ''

try:
    os.makedirs(destination)
except:
    pass

formats_to_convert = ['.m4a']

if os.path.isdir(source):
    for (dirpath, dirnames, filenames) in os.walk(source):
        for filename in filenames:
            if filename.endswith(tuple(formats_to_convert)):
                filepath = dirpath + '/' + filename
                (path, file_extension) = os.path.splitext(filepath)
                file_extension_final = file_extension.replace('.', '')
                try:
                    track = AudioSegment.from_file(filepath, file_extension_final)
                    wav_path = path.replace(source, destination) + '.wav'
                    print('CONVERTING: ' + str(filepath))
                    try:
                        os.makedirs(os.path.dirname(wav_path))
                    except:
                        pass
                    file_handle = track.export(wav_path, format='wav')
                except:
                    print("ERROR CONVERTING " + str(filepath))
else:
    print('source does not exist!')
