import numpy as np
from Generate_MFCC import *


audio_file = 'audio/per0003.wav'

mfcc = getMfccs(audio_file)

print('rows:', len(mfcc))
print('cols:', len(mfcc[0]))
# print(mfcc)

mfcc_reshaped = mfcc.reshape(1, 10, 70)
print(mfcc_reshaped)
print('wrap:', len(mfcc_reshaped))
print('rows:', len(mfcc_reshaped[0]))
print('cols:', len(mfcc_reshaped[0][0]))

