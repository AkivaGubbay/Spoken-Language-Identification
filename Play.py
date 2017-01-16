import numpy as np
from Generate_MFCC import *

a = [[1, 2, 3], [4, 5, 6]]

print('a[0][3/4] = ', a[0][3/4])

'''
a = a.reshape(1, 1, 6)
print(a, '\n')
'''


mfcc = getMfccs('audio/per0003.wav')
mfcc = mfcc[0:2, 0:3]
print('before reshape:\n', mfcc, '\n')
mfcc = mfcc.reshape(1, 1, 6)
print('\nafter reshape:\n', mfcc, '\n')