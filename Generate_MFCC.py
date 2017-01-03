import librosa
from Parameters import *
import numpy as np

def getMfccs(audio_file_name):
    wave, sr = librosa.load(audio_file_name, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    return mfcc[0:NUM_OF_COEFF, TIME_S: TIME_F]
    '''
    # My code:
    shorter_mfcc = []
    for i in range(0, NUM_OF_COEFF):   # number of coeff.
        tmp = []
        for j in range(TIME_S, TIME_F):  # Range on time in audio.
            try:
                tmp.append(mfcc[i, j])
            except:
                print('My FRROR mfcc no value at this range.')
                tmp.append(1)
        shorter_mfcc.append(tmp)
    return shorter_mfcc
    '''



'''
audio_file_name = 'audio/per0014.wav'

mfcc = getMfccs(audio_file_name)
print(mfcc)
'''

