# https://github.com/manashmandal/DeadSimpleSpeechRecognizer/blob/master/preprocess.py

import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import sys

DATA_PATH = "/Users/hyung.lee/cs229fall2019/speech_commands_tenlabels/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    alldir = os.listdir(path)
    filesToRemove = [
        'testing_list.txt',
        'validation_list.txt',
        'LICENSE','README.md',
        '.DS_Store',
        'valid_audio',
        'test_audio',
        'train_audio',
        '_background_noise_',
        'cnnmodel.h5',
        'cnnmodel2.h5',
        'cnnmodel3.h5']
    for f in filesToRemove:
        if alldir.__contains__(f):
            alldir.remove(f)
    labels = alldir
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = np.array(wave[::3])
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=20)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return np.sum(mfcc, axis=1)/11

def save_data_to_array(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        print(mfcc_vectors[0].shape)
        np.save(label + '.npy', mfcc_vectors)


def get_train_test_valid(split_ratio=0.6, random_state=42, path=DATA_PATH):
    # Get available labels
    labels, indices, _ = get_labels(path)

    # Getting first arrays
    X = np.load('models/' + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load('models/' + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)
    X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=(0.2), random_state=random_state, shuffle=True)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, test_size=(0.2), random_state=random_state, shuffle=True)
    return (X_train, X_validate, X_test, y_train, y_validate, y_test)


def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data

def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]

def main(path):
    save_data_to_array(path, max_len=11)


if __name__ == "__main__":
    main(sys.argv[1])

