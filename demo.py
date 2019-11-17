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
import csv

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
    iteration=0
    n_fft = 4096
    num_mfcc = 12
    hop_length = n_fft // 2
    max_time_stamp = 44    #most files are 32k except noise which can be upto 3M
    # max_time_stamp = 0

    for fileName in fileList:
        updated_feature_array = np.zeros((num_mfcc, max_time_stamp))
        iteration += 1
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[0]
        # print(label)
        # map label to integer
        label_int = maplabel_to_int(label)

        # load wav file
        y,sample_rate = librosa.load(dir+fileName)
        # convert to mfcc
        feature = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=num_mfcc)
        # feature is a numpy array
        # print(type(feature))
        # print(feature.shape)

        time_stamp = feature.shape[1]
        # print(feature.shape)
        # print(updated_feature_array.shape)

        # this is added since  time stamps vary between wav files
        pad = 0

        if time_stamp < max_time_stamp:
            # current file is shorter than the max. duration, pad feature vector again for remaining time duration
            for i in range(time_stamp):
                updated_feature_array[:,i] = feature[:,i]
            for j in range(max_time_stamp - time_stamp):
                updated_time_stamp = time_stamp + j
                updated_feature_array[:, updated_time_stamp] = feature[:,pad]
                pad += 1
                if pad == time_stamp:
                    pad = 0
                # print('Pad index is', pad)
                # updated_feature_array[:, updated_time_stamp] = 0
        else:
            # current time is greater than max. duration, copy only portion of file upto max. time stamp ignore the rest
            for i in range(max_time_stamp):
                updated_feature_array[:,i] = feature[:,i]

        if (iteration % 500 == 0):
            print('i = ', iteration, 'feature shape = ', updated_feature_array.shape)

        if i < 1:   # this is set to higher number is want to generate plots, set to 1 if want to skip plot
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

        del feature # to save memory

        if label_int not in dataset.keys():
            dataset[label_int] = []
            dataset[label_int].append(updated_feature_array)
        else:
            exist_feature = dataset[label_int]
            exist_feature.append(updated_feature_array)
            dataset[label_int] = exist_feature

    # dataset[label_int] = np.array(dataset[label_int])
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
    startprobPrior = np.array([0.3, 0.2, 0.1, 0.2, 0.2],dtype=np.float)

    for label in dataset.keys():

        print('Label is ', label)
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, transmat_prior=transmatPrior, startprob_prior=startprobPrior, covariance_type='diag', n_iter=10)
        trainData = dataset[label]

        # print(np.array(trainData[m])) # this has shape 12,44

        trainData_array = np.asarray(trainData)
        print('Original dimension is', trainData_array.shape)
        trainData_array = np.dstack(np.asarray(trainData))
        print('D stack dimension is', trainData_array.shape)

        trainData_array = np.vstack(trainData_array)
        print('V stack dimension is', trainData_array.shape)

        trainData_array = np.transpose(trainData_array)
        print('Final dimension is', trainData_array.shape)
        length = ((trainData_array.shape[0],))


        model.fit(trainData_array, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model

    return GMMHMM_Models

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
    trainDir = train_dirpath + '/'
    trainDataSet = buildDataSet(trainDir)
    # save trainDataset to csv file so dont have to rebuild the dataset
    with open('trainDataSet.csv', 'w') as f:
        for key in trainDataSet.keys():
            f.write("%s,%s\n" % (key, trainDataSet[key]))
    # # read csv file to dictionary
    # reader = csv.reader(open('trainDataSet.csv', 'r'))
    # trainDataSet = {}
    # for row in reader:
    #     label, v = row
    #     trainDataSet[label] = v

    print("Finished preparing the training data")
    hmmModels = train_GMMHMM(trainDataSet)
    print("Finished training of the GMM_HMM models using train data")

    validDir = validation_dirpath + '/'
    validDataSet = buildDataSet(validDir)
    print("Finished preparing the validation data")
    # save validDataset to csv file so dont have to rebuild the dataset
    with open('validDataSet.csv', 'w') as f:
        for key in validDataSet.keys():
            f.write("%s,%s\n" % (key, validDataSet[key]))
    # # read csv file to dictionary
    # reader = csv.reader(open('validDataSet.csv', 'r'))
    # validDataSet = {}
    # for row in reader:
    #     label, v = row
    #     validDataSet[label] = v

    testDir = test_dirpath + '/'
    testDataSet = buildDataSet(testDir)
    print("Finished preparing the test data")
    # save testDataset to csv file so dont have to rebuild the dataset
    with open('testDataSet.csv', 'w') as f:
        for key in testDataSet.keys():
            f.write("%s,%s\n" % (key, testDataSet[key]))
    # # read csv file to dictionary
    # reader = csv.reader(open('testDataSet.csv', 'r'))
    # testDataSet = {}
    # for row in reader:
    #     label, v = row
    #     testDataSet[label] = v

    score_cnt = 0
    for label in testDataSet.keys():
        testData = testDataSet[label]
        # print(label)

        # testData_array = np.asarray(testData)
        testData_array = np.dstack(np.asarray(testData))
        print('Original dimension of test array is', testData_array.shape)
        testData_array = np.vstack(testData_array)
        print('After vstack dimension of test array is', testData_array.shape)
        testData_array = np.transpose(testData_array)
        print('After transponse dimension of test array is', testData_array.shape)
        length = ((testData_array.shape[0],))

        scoreList = {}
        for model_label in hmmModels.keys():
            # print('Model Label is ', model_label)
            model = hmmModels[model_label]
            score = model.score(testData_array, lengths=length)
            # print('score is ', score)
            scoreList[model_label] = score
            # print('score list is ', scoreList)

        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate on test set is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")

    score_cnt = 0
    for label in validDataSet.keys():
        validData = validDataSet[label]

        # validData_array = np.asarray(validData)
        validData_array = np.dstack(np.asarray(validData))

        validData_array = np.vstack(validData_array)

        validData_array = np.transpose(validData_array)
        length = ((validData_array.shape[0],))

        scoreList = {}
        for model_label in hmmModels.keys():
            # print('Model Label is ', model_label)
            model = hmmModels[model_label]
            score = model.score(validData_array, lengths=length)
            # print('score is ', score)
            scoreList[model_label] = score
            # print('score list is ', scoreList)

        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate on validation set is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")

if __name__ == '__main__':
    main()