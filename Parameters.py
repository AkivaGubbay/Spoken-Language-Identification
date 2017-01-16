
# /////////////////////////
NUM_OF_COEFF = 10   # 10
TIME_S = 30     # 30
TIME_F = 110    # 100
current_batch = 0
all_mfcc_vectors = []
all_audio_files = []
num_of_audio_in_language = 100  # 100
validation = int(0.15 * num_of_audio_in_language)
# ///////////////

learning_rate = 0.00001  # without dropout: 0.000001  # with dropout: 0.0000001  # default learning rate of 0.001(for Adam).
hm_epochs = 600     # without dropout: 600
n_classes = 4
batch_size = 1
chunk_size = (TIME_F - TIME_S)
n_chunks = NUM_OF_COEFF
rnn_size = 128  # make this bigger
up_to_subfile = []


num_of_batches_in_language = int(num_of_audio_in_language / batch_size)