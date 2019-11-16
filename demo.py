'''
Created on 11/15/2019
@author: apatil
used this github repo as starting point
https://github.com/wblgers/hmm_speech_recognition_demo/blob/master/demo.py
'''
from __future__ import print_function
import warnings
import os

import matplotlib.pyplot as plt
import shutil
from hmmlearn import hmm
import numpy as np
import librosa
import librosa.display

## currently unused libraries
# from scipy.io import wavfile
# from speechpy import feature
# import scipy

np.random.seed(42)

warnings.filterwarnings('ignore')

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def  maplabel_to_int(label):
    switcher = {
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
        'forward': 10,
        'backward': 11,
        'up': 12,
        'down': 13,
        'yes': 14,
        'no': 15,
        'on': 16,
        'off': 17,
        'right': 18,
        'left': 19,
        'go': 20,
        'stop': 21,
        'follow': 22,
        'bed': 23,
        'bird': 24,
        'cat': 25,
        'dog': 26,
        'happy': 27,
        'house': 28,
        'learn': 29,
        'marvin': 30,
        'sheila': 31,
        'tree': 32,
        'visual': 33,
        'wow': 34,
        '': 35,  # this stands for noise

    }
    return switcher.get(label, "Invalid label")


def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    i=0
    n_fft = 4096
    hop_length = n_fft // 2

    for fileName in fileList:
        i += 1
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[0]
        # print(label)
        # map label to integer
        label_int = maplabel_to_int(label)

        # load wav file
        y,sample_rate = librosa.load(dir+fileName)
        # convert to mfcc
        feature = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=12)

        if i < 1:   # this is set to higher number is want to generate plots
            # save mfcc feature plot
            plt.figure()
            ax = plt.subplot(2, 1, 1)
            plt.title('MFCC')
            librosa.display.specshow(feature, x_axis='time')
            plt.colorbar()

            # save spectrum
            plt.subplot(2, 1, 2, sharex=ax)
            plt.title('PCEN spectrum')
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
            # Compute pcen on the magnitude spectrum.
            # We don't need to worry about initial and final filter delays if
            # we're doing everything in one go.
            P = librosa.pcen(np.abs(D), sr=sample_rate, hop_length=hop_length)
            # First, plot the spectrum
            librosa.display.specshow(P, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
            plt.tight_layout()
            file_name = 'mfcc{}_spectrum_{}.pdf'.format('_', label)
            save_path = os.path.join('.', file_name)
            plt.savefig(save_path)

        if label_int not in dataset.keys():
            dataset[label_int] = []
            dataset[label_int].append(feature)
        else:
            exist_feature = dataset[label_int]
            exist_feature.append(feature)
            dataset[label_int] = exist_feature

    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5              # number of components
    GMM_mix_num = 3             # was 3 in the online example that trained numbers 1 through 10, 10 observations per number
    tmp_p = 1.0/(states_num-2)

    # dim of transmatPrior is n_components x n_components
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)

    # dim of startprobPrior is n_components
    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, transmat_prior=transmatPrior, startprob_prior=startprobPrior, covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
            print(length[m])
            # print(m)

            
        # print(np.array(trainData[m])) # this has shape 12,44
        # trainData = np.vstack(np.asarray(trainData))
        print(np.asarray(trainData).shape)

        model.fit(trainData, lengths=length)  # get optimal parameters
        # model.fit(trainData)
        GMMHMM_Models[label] = model
    return GMMHMM_Models


    # for label in dataset.keys():
    #     model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
    #                        transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
    #                        covariance_type='full', n_iter=10)
    #
    #     trainData_array = np.array(dataset[label])
    #     num_obs = np.zeros([trainData_array.shape[0], 1], dtype=np.int)
    #
    #     for m in range(len(dataset[label])):    # this corresponds to # of observations
    #         num_obs[m] = label
    #
    #     # num_obs.reshape(-1,1)
    #     print('Shape of label is ', num_obs.shape)
    #     print('Shape of data is', trainData_array.shape)
    #     print('Label is ', num_obs)
    #     print('Data is', trainData_array)
    #
    #     model.fit(trainData_array, lengths=num_obs)  # get optimal parameters
    #     GMMHMM_Models[label] = model
    # return GMMHMM_Models


def main():
    dir = '/Users/amitapatil/Desktop/ACP/CS229_MachineLearning/Project/Dataset/speech_commands_v0.02'
    alldir = os.listdir(dir)

    train_dirname = 'train_audio'
    train_dirpath = os.path.join(dir, train_dirname)
    if not os.path.exists(train_dirpath):
        os.makedirs(train_dirpath)

    test_dirname = 'test_audio'
    test_dirpath = os.path.join(dir, test_dirname)
    if not os.path.exists(test_dirpath):
        os.makedirs(test_dirpath)

    validation_dirname = 'valid_audio'
    validation_dirpath = os.path.join(dir,validation_dirname)
    if not os.path.exists(validation_dirpath):
        os.makedirs(validation_dirpath)

    # ## THIS PORTION OF CODE CAN BE COMMENTED OUT AFTER TRAIN/VALIDATION/TEST SETS ARE CREATED
    ## #there maybe better way  to remove these sub dir
    # alldir.remove('testing_list.txt')
    # alldir.remove('validation_list.txt')
    # alldir.remove('LICENSE')
    # alldir.remove('README.md')
    # alldir.remove('.DS_Store')
    #
    # allFiles = []
    # for i in range(len(alldir)):
    #     pathname = os.path.join(dir, alldir[i])
    #     filename = os.listdir(pathname)
    #     for j in filename:
    #         filename_with_dir = os.path.join(alldir[i],j)
    #         allFiles.append(filename_with_dir)
    #
    #
    # validation_file = open(os.path.join(dir,"validation_list.txt"), "r")
    # validation_list = validation_file.readlines()
    # validation_file.close()
    # # stripping \n from each element of validation list
    # validation_list = map(lambda s: s.strip(), validation_list)
    # print(validation_list)
    #
    # test_file = open(os.path.join(dir,"testing_list.txt"), "r")
    # testing_list = test_file.readlines()
    # test_file.close()
    # # stripping \n from each element of testing list
    # testing_list = map(lambda s: s.strip(), testing_list)
    # print(testing_list)

    # # removes validation wavefile from all files
    # for item in validation_list:
    #     if item in allFiles:
    #         allFiles.remove(item)
    # # removes testing wavefile from all files
    # for item in testing_list:
    #     if item in allFiles:
    #         allFiles.remove(item)

    # # saves validation files in
    # for items in validation_list:
    #     newname = items.split('/')[0] + '_' + items.split('/')[-1]
    #     file1 = os.path.join(os.path.join(dir,items.split('/')[0]),items.split('/')[-1])
    #     file2 = os.path.join(validation_dirpath, newname)
    #     print(file2)
    #     shutil.copy(file1,file2)
    # print('Done with writing wav files in validation folder')
    #
    # for items in testing_list:
    #     newname = items.split('/')[0] + '_' + items.split('/')[-1]
    #     file1 = os.path.join(os.path.join(dir,items.split('/')[0]),items.split('/')[-1])
    #     file2 = os.path.join(test_dirpath, newname)
    #     print(file2)
    #     shutil.copy(file1,file2)
    # print('Done with writing wav files in test folder')

    # for items in allFiles:
    #     newname = items.split('/')[0] + '_' + items.split('/')[-1]
    #     file1 = os.path.join(os.path.join(dir, items.split('/')[0]), items.split('/')[-1])
    #     file2 = os.path.join(train_dirpath, newname)
    #     shutil.copy(file1,file2)
    # print('Done with writing wav files in train folder')

    ## THIS PORTION OF CODE CAN BE COMMENTED OUT AFTER TRAIN/VALIDATION/TEST SETS ARE CREATED

    ## work with validation directory 1st to debug code since train set is too big
    # trainDir = train_dirpath + '/'
    # print(trainDir)
    # trainDataSet = buildDataSet(trainDir)
    # print("Finish preparing the training data")
    # hmmModels = train_GMMHMM(trainDataSet)
    # print("Finish training of the GMM_HMM models using train data")

    validDir = validation_dirpath + '/'
    validDataSet = buildDataSet(validDir)
    print("Finish preparing the validation data")

    hmmModels = train_GMMHMM(validDataSet)
    print("Finish training of the GMM_HMM models using validation data")
    #
    # testDir = test_dirpath + '/'
    # testDataSet = buildDataSet(testDir)
    # print("Finish preparing the test data")
    #
    # # this codde is purely for debugging
    # # label = 'backward'
    # # label = 'bed'
    # # label = ''
    # # label_int = maplabel_to_int(label)
    #
    # score_cnt = 0
    # for label in testDataSet.keys():
    #     feature = testDataSet[label]
    #     scoreList = {}
    #     for model_label in hmmModels.keys():
    #         model = hmmModels[model_label]
    #         score = model.score(feature[0])
    #         scoreList[model_label] = score
    #     predict = max(scoreList, key=scoreList.get)
    #     print("Test on true label ", label, ": predict result label is ", predict)
    #     if predict == label:
    #         score_cnt+=1
    # print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")

if __name__ == '__main__':
    main()