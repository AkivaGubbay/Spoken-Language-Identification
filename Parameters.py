hm_epochs = 50   # A lot more..
n_classes = 4
batch_size = 1
chunk_size = 4  # length of time
n_chunks = 20   # number of mfcc coeff
rnn_size = 128  # make this bigger

# /////////////////////////Mine:

current_batch = 0
all_mfcc_vectors = []
num_of_audio_in_language = 3
num_of_batches_in_language = int(num_of_audio_in_language / batch_size)