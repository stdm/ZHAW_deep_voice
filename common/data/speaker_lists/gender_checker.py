"""

"""
import sys

with open(sys.argv[1], 'rb') as f:
    file = f.readlines()


male_speakers = 0
total_speakers = 0

for speaker_name in file:
    speaker_name = bytes.decode(speaker_name.rstrip())
    if speaker_name is not '':
        total_speakers = total_speakers + 1
    if speaker_name.startswith('M'):
        male_speakers = male_speakers + 1

male_percentage = (male_speakers/total_speakers)*100
female_percentage = 100-male_percentage

print('There are ', male_percentage, '% male and ', female_percentage, '% female speakers in this list. \n')
if round(male_percentage) == 70:
    print('The list is stratified!')
