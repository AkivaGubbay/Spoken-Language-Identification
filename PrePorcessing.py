import os

import numpy

from Parameters import *
from Generate_MFCC import getMfccs
import numpy as np

def fun1(parent_directory, _from, _to):
    global all_audio_files
    languages = os.listdir(parent_directory)
    all_audio_files = []
    print(languages)
    for lang in languages:
        language_files = []
        lang_dir = parent_directory + '/' + lang
        # print(lang_dir)
        for audio_file in os.listdir(lang_dir)[_from:_to]:   # where is the mfcc????
            audio_dir = lang_dir + '/' + audio_file
            language_files.append(audio_dir)
            # print(language_files)
        all_audio_files.append(language_files)

'''
directory = r'/media/akiva/Seagate Backup Plus Drive/voxforge/parent'
audio_to_mfccs(directory, 0, 2)
for i in range(0,4):
    print(all_audio_files[i])
'''
def build_up_to_subfile():
    global  up_to_subfile
    up_to_subfile = [True]
    for i in range(1, n_classes):
        up_to_subfile.append(False)

build_up_to_subfile()

def fun2(audio_in_sub_file):
    global current_batch, up_to_subfile, all_audio_files

    # Determine which sub-file we are up too.
    # So I can pass real classification.
    subfile = 0
    for i in range(3, 0, -1):
        if up_to_subfile[i] is True:
            subfile = i
            break

    # Case: finished a sub-file.
    if current_batch >= audio_in_sub_file:
        # Case: finished all audio files.
        if up_to_subfile[3] is True:
            # Start from beginning for next epoch.
            current_batch = 0
            subfile = 0
            up_to_subfile = [True, False, False, False]
        # Case: just finished a certain sub-file:
        else:
            for i in range(0, 4):
                if up_to_subfile[i] is False:
                    # Next sub-file.
                    up_to_subfile[i] = True
                    subfile = i
                    # Start the count again.
                    current_batch = 0
                    break


    # I need to pass epoch_x, epoch_y as numpy arrays:
    # batch_y = real classification of audio.
    try:
        epoch_y = numpy.zeros(shape=(batch_size, 4))
        epoch_y[0][subfile] = 1
    except:
        print('my Error reshaping batch_y.')

    epoch_x = getMfccs(all_audio_files[subfile][current_batch])

    '''
    # printing info
    print('current_batch:', current_batch)
    print('subfile:', subfile)
    print('up_to_subfile:', up_to_subfile, '\n')
    '''
    # update 'current_batch' for next time.
    current_batch += 1
    return epoch_x, epoch_y






# ===   These function do the same thing but calculate all the mfcc's and holds them all in one list.     ==============
#====   Tis is much faster but when dealing with lots of lifes I might get memory problems,               ==============


def audio_to_mfccs(directory, _from, _to):
    global all_mfcc_vectors

    all_mfcc_vectors = []
    parent = directory
    languages = os.listdir(parent)
    print(languages)
    for lang in languages:
        lang_dir = parent + '/' + lang
        # print('I am checking language: ', lang)
        # print('from: ', _from, '\tto:', _to)
        # print('list of audio to learn: ', os.listdir(lang_dir)[_from:_to])
        for audio_file in os.listdir(lang_dir)[_from:_to]:
            # print('Going over audio file: ', audio_file)
            audio_dir = lang_dir + '/' + audio_file
            mfcc = getMfccs(audio_dir)
            all_mfcc_vectors.append(mfcc)
    print('finished calculating mfcc')



# up_to_subfile = [True, False, False, False]

def next_batch(audio_in_sub_file):
    global current_batch, up_to_subfile, all_mfcc_vectors, n_classes

    # Determine which sub-file we are up too.
    # So I can pass real classification.
    subfile = 0
    for i in range(n_classes - 1, 0, -1):    # 3
        if up_to_subfile[i] is True:
            subfile = i
            break

    # Case: finished a sub-file.
    if current_batch >= audio_in_sub_file:
        # Case: finished all audio files.
        if up_to_subfile[n_classes - 1] is True:    # 3
            # Start from beginning for next epoch.
            current_batch = 0
            subfile = 0
            build_up_to_subfile()
        # Case: just finished a certain sub-file:
        else:
            for i in range(0, n_classes):    # 4
                if up_to_subfile[i] is False:
                    # Next sub-file.
                    up_to_subfile[i] = True
                    subfile = i
                    # Start the count again.
                    current_batch = 0
                    break

    # Index in 'all_mfcc_vectors':
    start = (subfile * audio_in_sub_file) + current_batch
    '''
    # printing info
    print('current_batch:', current_batch)
    print('subfile:', subfile)
    print('start:', start)
    print('up_to_subfile:', up_to_subfile, '\n')
    '''
    # update 'current_batch' for next time.
    current_batch += 1
    # I need to pass epoch_x, epoch_y as numpy arrays:
    # batch_y = real classification of audio.
    try:
        epoch_y = numpy.zeros(shape=(batch_size, n_classes))     # 4
        # [0][subfile] because it's a list that holds one list,
        # that is the hot vector.
        epoch_y[0][subfile] = 1
    except:
        print('my Error reshaping batch_y.')
    epoch_x = all_mfcc_vectors[start]
    return epoch_x, epoch_y






