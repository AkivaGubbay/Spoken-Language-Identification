import librosa
from Parameters import *
import numpy as np


def getMfccs(audio_file_name):
    wave, sr = librosa.load(audio_file_name, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    try:
        mfcc = mfcc[0:NUM_OF_COEFF, TIME_S: TIME_F]
    except:
        print('audio file: ', audio_file_name, ' was not long enuogh.')
        return None
    return mfcc



'''
def getMfccs(audio_file_name):
    wave, sr = librosa.load(audio_file_name, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    return mfcc[0:NUM_OF_COEFF, TIME_S: TIME_F]
'''

'''
audio_file_name = 'audio/per0014.wav'

mfcc = getMfccs(audio_file_name)
print(mfcc)
'''

