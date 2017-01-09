
# /////////////////////////
NUM_OF_COEFF = 10   # 10
TIME_S = 30     # 30
TIME_F = 180    # 100
current_batch = 0
all_mfcc_vectors = []
all_audio_files = []
num_of_audio_in_language = 100  # 100
validation = int(0.15 * num_of_audio_in_language)   # 15
# ///////////////

input_dropout = output_dropout = 0.9
learning_rate = 0.0001
hm_epochs = 300   # 40 # A lot more..                                     # FOR 500 AUDIO'S MAKE MUCH MORE THEN 40 EPOCHS
n_classes = 4
batch_size = 1
chunk_size = (TIME_F - TIME_S)  # length of time
n_chunks = NUM_OF_COEFF   # number of mfcc coeff
rnn_size = 128  # make this bigger



num_of_batches_in_language = int(num_of_audio_in_language / batch_size)