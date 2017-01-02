import os
from Parameters import *
from Generate_MFCC import getMfccs

# if I only calculate the vectors in my batch I wont need to hold all the memory at once.

def audio_to_mfccs():
    global all_mfcc_vectors, num_of_audio_in_language

    parent = r'/media/akiva/Seagate Backup Plus Drive/voxforge/parent'  # The 'r' is to prevent white spaces.
    languages = os.listdir(parent)
    print(languages)
    for lang in languages:
        lang_dir = parent + '/' + lang
        for audio_file in os.listdir(lang_dir)[:num_of_audio_in_language]:
            audio_dir = lang_dir + '/' + audio_file
            mfcc = getMfccs(audio_dir)
            # print(mfcc)
            all_mfcc_vectors.append(mfcc)
            # break
    #print('all_mfcc_vectors rows:', len(all_mfcc_vectors))
    #print('all_mfcc_vectors cols:', len(all_mfcc_vectors[0]))
    #print('all_mfcc_vectors')
    #for line in all_mfcc_vectors:
        #print(line)
    # print('all_mfcc_vectors', all_mfcc_vectors)


def next_batch():
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



