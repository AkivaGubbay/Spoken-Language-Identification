
# /////////////////////////Mine:
NUM_OF_COEFF = 10   # 10
TIME_S = 30     # 30
TIME_F = 80    # 100
current_batch = 0
all_mfcc_vectors = []
num_of_audio_in_language = 100  # 41
validation = 15   # 15
# ///////////////

hm_epochs = 30   # A lot more..
n_classes = 4
batch_size = 1
chunk_size = (TIME_F - TIME_S)  # length of time
n_chunks = NUM_OF_COEFF   # number of mfcc coeff
rnn_size = 128  # make this bigger



num_of_batches_in_language = int(num_of_audio_in_language / batch_size)