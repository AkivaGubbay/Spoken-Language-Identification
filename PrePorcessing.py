import os
from Generate_MFCC import getMfccs

parent = r'/media/akiva/Seagate Backup Plus Drive/voxforge/parent'  # The 'r' is to prevent white spaces.
languages = os.listdir(parent)
print(languages)

def audio_to_mfccs():

    all_mfcc_vectors = []
    for lang in languages:
        lang_dir = parent + '/' + lang
        print(lang_dir)
        for audio_file in os.listdir(lang_dir)[:10]:
            audio_dir = lang_dir + '/' + audio_file
            mfcc = getMfccs(audio_dir)
            # print(mfcc)
            all_mfcc_vectors.append(mfcc)
            # break
    print('rows:', len(all_mfcc_vectors))
    print('cols:', len(all_mfcc_vectors[0]))
    print('all_mfcc_vectors', all_mfcc_vectors)

audio_to_mfccs()

# Make one array of all files.
# But then I need to deal with the classification.
# languages = os.listdir(parent)
# lang_dir = parent + '/' + lang

# list_of_all_audio_files =




# where_i_was = 0
# def next_batch(batch_size):



