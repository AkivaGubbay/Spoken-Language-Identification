import os

import numpy

from Parameters import *
from Generate_MFCC import getMfccs
import numpy as np

# if I only calculate the vectors in my batch I wont need to hold all the memory at once.

def audio_to_mfccs(directory, _from, _to):
    global all_mfcc_vectors, num_of_audio_in_language

    all_mfcc_vectors = []
    parent = directory
    languages = os.listdir(parent)
    print(languages)
    for lang in languages:
        lang_dir = parent + '/' + lang
        print('I am checking language: ', lang)
        print('from: ', _from, '\tto:', _to)
        print('list of audio to learn: ', os.listdir(lang_dir)[_from:_to])
        for audio_file in os.listdir(lang_dir)[_from:_to]:
            print('Going over audio file: ', audio_file)
            audio_dir = lang_dir + '/' + audio_file
            mfcc = getMfccs(audio_dir)
            all_mfcc_vectors.append(mfcc)
    print('all_mfcc_vectors:', all_mfcc_vectors)
up_to_subfile = [True, False, False, False]

def next_batch(audio_in_sub_file):
    global current_batch, up_to_subfile, all_mfcc_vectors

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

    # Index in 'all_mfcc_vectors':
    start = (subfile * audio_in_sub_file) + current_batch

    # printing info
    print('current_batch:', current_batch)
    print('subfile:', subfile)
    print('start:', start)
    print('up_to_subfile:', up_to_subfile, '\n')

    # update 'current_batch' for next time.
    current_batch += 1
    # I need to pass epoch_x, epoch_y as numpy arrays:
    # batch_y = real classification of audio.
    try:
        epoch_y = numpy.zeros(shape=(batch_size, 4))
        # [0][subfile] because it's a list that holds one list,
        # that is the hot vector.
        epoch_y[0][subfile] = 1
    except:
        print('my Error reshaping batch_y.')
    epoch_x = all_mfcc_vectors[start]
    return epoch_x, epoch_y



# audio_to_mfccs()
# print(next_batch())


'''
    global current_batch, batch_size, all_mfcc_vectors

    if (current_batch + batch_size) >= len(all_mfcc_vectors):
        return "My ERROR: to many batches"
    start = current_batch
    end = current_batch + batch_size
    real_classification = classification_in_vector()
    current_batch += batch_size
    return all_mfcc_vectors[start:end], real_classification


# maybe one hot vector..
def classification_in_vector():
    global current_batch

    if (current_batch > 0) and (current_batch < num_of_audio_in_language):
        return [1, 0, 0, 0]
    if (current_batch > num_of_audio_in_language) and (current_batch < 2*num_of_audio_in_language):
        return [0, 1, 0, 0]
    if (current_batch > 2*num_of_audio_in_language) and (current_batch < 3 * num_of_audio_in_language):
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]


# audio_to_mfccs()

# Make one array of all files.
# But then I need to deal with the classification.
# languages = os.listdir(parent)
# lang_dir = parent + '/' + lang

# list_of_all_audio_files =




# where_i_was = 0
# def next_batch(batch_size):


'''
