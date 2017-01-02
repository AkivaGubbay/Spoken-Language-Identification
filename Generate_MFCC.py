import librosa

def getMfccs(audio_file_name):
    wave, sr = librosa.load(audio_file_name, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    # My code:
    shorter_mfcc = []
    for i in range(len(mfcc)):
        tmp = []
        for j in range(20, 24):  # Range on time in audio.
            tmp.append(mfcc[i, j])
        shorter_mfcc.append(tmp)
    return shorter_mfcc


'''
audio_file_name = 'audio/per0014.wav'

mfcc = getMfccs(audio_file_name)
print('mfcc:')
for line in mfcc:
    print(line,', ')
'''


