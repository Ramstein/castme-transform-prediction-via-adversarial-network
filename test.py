import librosa, os, matplotlib.pyplot as plt
SAMPLE_RATE = 44100

files = ['Conv17.wav', 'Conv16.wav', 'Conv15.wav']
for file in files:
    fname = os.path.join(r'C:\Conv\wavs', file)   # Hi-hat
    wav, _ = librosa.core.load(fname, sr=SAMPLE_RATE)
    wav = wav[:2*44100]



    mfcc = librosa.feature.mfcc(wav, sr = SAMPLE_RATE, n_mfcc=40)
    print(mfcc.shape)

    plt.imshow(mfcc, cmap='hot', interpolation='nearest')
    plt.show()