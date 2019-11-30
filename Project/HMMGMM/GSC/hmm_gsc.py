'''
Created on 11/15/2019
@author: apatil
'''
from __future__ import print_function
import warnings
import os

import matplotlib.pyplot as plt
import shutil
from hmmlearn import hmm
from hmmlearn.base import ConvergenceMonitor
import numpy as np
import librosa
import librosa.display
import csv
import pickle

import scipy
from scipy.io import wavfile

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools

import sys

# np.random.seed(42)

warnings.filterwarnings('ignore')

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def  maplabel_to_int(label):

    ## When you pass label to the maplabel_to_int function, it is looked up against the switcher dictionary mapping.
    ## If a match is found, the associated value is returned
    ## else a default string (‘Invalid label’) is returned

    switcher = {
        # these are core words (total 22), we train 1 HMM model for each of these core words
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'yes': 10,
        'no': 11,
        'up': 12,
        'down': 13,
        'left': 14,
        'right': 15,
        'on': 16,
        'off': 17,
        'stop': 18,
        'go': 19,
        'forward': 20,
        'backward': 21,

        # these are (total 13) auxillary words, we train 1 HMM model for all these auxillary words
        'follow': 22,
        'bed': 22,
        'bird': 22,
        'cat': 22,
        'dog': 22,
        'happy': 22,
        'house': 22,
        'learn': 22,
        'marvin': 22,
        'sheila': 22,
        'tree': 22,
        'visual': 22,
        'wow': 22,

        # this is for background noise, we train 1 HMM model - this model corresponds to background noise
        '': 35,  # this stands for noise

    }
    return switcher.get(label, "Invalid label")

def generatePlots(dir):

    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']


    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[0]

        y, sr = librosa.load(dir + fileName)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        plt.figure(figsize=(8, 4))
        plt.subplot(2, 1, 1)
        plt.title('Mel-frequency spectrogram')
        librosa.display.specshow(D)
        plt.colorbar(format='%+2.0f dB')
        # plt.tight_layout()

        plt.subplot(2, 1, 2)
        librosa.display.waveplot(y, sr=sr, x_axis='time')
        plt.title('Wave plot')

        plot_name = 'spectrogram_wave_' + label + '.pdf'
        print(plot_name)
        save_path = os.path.join('.', plot_name)
        plt.savefig(save_path)

    return

def plot_confusion_matrix(cm, classes,normalize=True,title=None,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax = ax)
    ax.set(title=title,ylabel='True label',xlabel='Predicted label')


def buildDataSet(dir):

    "reads all wave files and return list of 2D features for each label"
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    num_mfcc = 13

    for fileName in fileList:
        # get label for the wav file
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[0]

        # label_int = maplabel_to_int(label)

        # load wav file
        y,sample_rate = librosa.load(dir+fileName)
        # convert to mfcc
        feature = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=num_mfcc)

        # if label_int not in dataset.keys():
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)

        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature

    return dataset


def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 20
    GMM_mix_num = 3
    threshold = 0.00001
    n_iterations = 20

    transmatPrior = np.ones((states_num,states_num))/states_num
    startprobPrior = np.ones(states_num)/states_num


    for label in dataset.keys():

        print('Label is ', label)

        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=n_iterations, tol=threshold, verbose=True)

        print('Convergence monitor for ' + str(label) + str(sys.stderr))

        trainData = dataset[label]  # this is list of 2D array
        trainData_array = np.transpose(np.concatenate(trainData,axis=1))
        length = ((trainData_array.shape[0],))

        model.fit(trainData_array, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model

    return GMMHMM_Models

def main():


    dir = '/Users/amitapatil/Desktop/ACP/CS229_MachineLearning/Project/Dataset/speech_commands_v0.02'
    alldir = os.listdir(dir)

    # ## work with validation directory 1st to debug code since train set is too big
    train_dirname = 'train_audio_subset'
    train_dirpath = os.path.join(dir, train_dirname)
    if not os.path.exists(train_dirpath):
        os.makedirs(train_dirpath)
    trainDir = train_dirpath + '/'
    trainDataSet = buildDataSet(trainDir)

    print("Finished preparing the training data")
    hmmModels = train_GMMHMM(trainDataSet)
    print("Finished training of the GMM_HMM models using train data")
    # saving the model
    with open("hmmgmm_model.pkl", "wb") as file: pickle.dump(hmmModels, file)


    validation_dirname = 'valid_audio_subset'
    validation_dirpath = os.path.join(dir,validation_dirname)
    if not os.path.exists(validation_dirpath):
        os.makedirs(validation_dirpath)

    validDir = validation_dirpath + '/'
    validDataSet = buildDataSet(validDir)
    print("Finished preparing the validation data")

    hmmModels = pickle.load(open("hmmgmm_model.pkl", "rb"))


    # this is checking how well prediction matches the label on train set
    num_files = 0
    score_cnt = 0

    y_train = []
    y_pred = []

    class_names = []

    for label in trainDataSet.keys():

        class_names.append(label)

        trainData = trainDataSet[label]

        for i in range(len(trainData)):
            num_files += 1

            data_array = np.transpose(trainData[i])
            length = ((data_array.shape[0],))
            # print(data_array.shape)

            scoreList = {}

            for model_label in hmmModels.keys():

                model = hmmModels[model_label]

                score = model.score(data_array, lengths=length)
                scoreList[model_label] = score


            predict = max(scoreList, key=scoreList.get)

            # print("Test on true label ", label, ": predict result label is ", predict)
            if predict == label:
                score_cnt += 1

            y_train.append(label)
            y_pred.append(predict)
    print("Final recognition rate on train set is", 100 * score_cnt / num_files)

    #plot confusion matrix
    y_train_array = np.array(y_train)
    y_pred_array = np.array(y_pred)
    np.set_printoptions(precision=2)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_train_array, y_pred_array)
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
    plot_name = 'ConfusionMatrix_Normalized_TrainSet.pdf'
    plt.savefig(os.path.join('.', plot_name))

    # Done checking how well prediction matches the label on train set

    num_files = 0
    score_cnt = 0

    y_valid = []
    y_pred = []

    class_names = []

    for label in validDataSet.keys():
        # print(label)
        class_names.append(label)

        validData = validDataSet[label]

        for i in range(len(validData)):
            num_files += 1

            data_array = np.transpose(validData[i])
            length = ((data_array.shape[0],))
            # print(data_array.shape)

            scoreList = {}

            for model_label in hmmModels.keys():

                model = hmmModels[model_label]

                score = model.score(data_array, lengths=length)
                scoreList[model_label] = score


            predict = max(scoreList, key=scoreList.get)

            print("Test on true label ", label, ": predict result label is ", predict)
            if predict == label:
                score_cnt += 1

            y_valid.append(label)
            y_pred.append(predict)
    print("Final recognition rate on validation set is", 100*score_cnt/num_files)


    #plot confusion matrix
    y_valid_array = np.array(y_valid)
    y_pred_array = np.array(y_pred)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_valid_array, y_pred_array)
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Normalized Confusion Matrix')
    plot_name = 'ConfusionMatrix_Normalized_ValidationSet.pdf'
    plt.savefig(os.path.join('.', plot_name))

    # generatePlots(validDir)

if __name__ == '__main__':
    main()
