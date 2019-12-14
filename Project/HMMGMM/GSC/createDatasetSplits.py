'''
'''
Created on 29/08/2018
@author: amita patil
'''
from __future__ import print_function
import warnings
import os
#from scikits.talkbox.features import mfcc
from speechpy import feature

from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np

import re
import hashlib
# from itertools import ifilterfalse
import shutil

warnings.filterwarnings('ignore')

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

# def which_set(filename, validation_percentage, testing_percentage):
#   """Determines which data partition the file should belong to.
#
#   We want to keep files in the same training, validation, or testing sets even
#   if new ones are added over time. This makes it less likely that testing
#   samples will accidentally be reused in training when long runs are restarted
#   for example. To keep this stability, a hash of the filename is taken and used
#   to determine which set it should belong to. This determination only depends on
#   the name and the set proportions, so it won't change as other files are added.
#
#   It's also useful to associate particular files as related (for example words
#   spoken by the same person), so anything after '_nohash_' in a filename is
#   ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
#   'bobby_nohash_1.wav' are always in the same set, for example.
#
#   Args:
#     filename: File path of the data sample.
#     validation_percentage: How much of the data set to use for validation.
#     testing_percentage: How much of the data set to use for testing.
#
#   Returns:
#     String, one of 'training', 'validation', or 'testing'.
#   """
#   base_name = os.path.basename(filename)
#   # We want to ignore anything after '_nohash_' in the file name when
#   # deciding which set to put a wav in, so the data set creator has a way of
#   # grouping wavs that are close variations of each other.
#   hash_name = re.sub(r'_nohash_.*$', '', base_name)
#   # This looks a bit magical, but we need to decide whether this file should
#   # go into the training, testing, or validation sets, and we want to keep
#   # existing files in the same set even if more files are subsequently
#   # added.
#   # To do that, we need a stable way of deciding based on just the file name
#   # itself, so we do a hash of that and then use that to generate a
#   # probability value that we use to assign it.
#
#
#   hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
#   # hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
#
#   percentage_hash = ((int(hash_name_hashed, 16) %
#                       (MAX_NUM_WAVS_PER_CLASS + 1)) *
#                      (100.0 / MAX_NUM_WAVS_PER_CLASS))
#   if percentage_hash < validation_percentage:
#     result = 'validation'
#   elif percentage_hash < (testing_percentage + validation_percentage):
#     result = 'testing'
#   else:
#     result = 'training'
#   return result

def extract_mfcc(full_audio_path):
    sample_rate, wave =  wavfile.read(full_audio_path)
    # print(sample_rate)
    #mfcc_features = mfcc(wave, nwin=int(sample_rate * 0.03), fs=sample_rate, nceps=12)[0]
    mfcc_features = feature.mfcc(wave, sampling_frequency=sample_rate,frame_length=0.020, num_cepstral=12)[0]
    return mfcc_features

# def maplabel(label):
    # converts label string to integer




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
    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[1]
        # label = tmp.split('_')[0]
        # print(label)
        # print(type(label))
        # label_int = maplabel_to_int(label)
        label_int = label
        # print(label_int)
        feature = extract_mfcc(dir+fileName)
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
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        # print(length)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
            print(length[m])
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models

def main():


    dir = '/Users/amitapatil/Desktop/ACP/CS229_MachineLearning/Project/Dataset/speech_commands_v0.02'
    alldir = os.listdir(dir)

    train_dirname = 'train_audio'
    train_dirpath = os.path.join(dir,train_dirname)
    if not os.path.exists(train_dirpath):
        os.makedirs(train_dirpath)

    test_dirname = 'test_audio'
    test_dirpath = os.path.join(dir,test_dirname)
    if not os.path.exists(test_dirpath):
        os.makedirs(test_dirpath)

    validation_dirname = 'valid_audio'
    validation_dirpath = os.path.join(dir,validation_dirname)
    if not os.path.exists(validation_dirpath):
        os.makedirs(validation_dirpath)

    ## THIS PORTION OF CODE CREATES TRAIN/VALIDATION/TEST SETS USING all wav files in dir
    # and testing_list and validation_list
    alldir.remove('testing_list.txt')
    alldir.remove('validation_list.txt')
    alldir.remove('LICENSE')
    alldir.remove('README.md')
    alldir.remove('.DS_Store')

    allFiles = []
    for i in range(len(alldir)):
        pathname = os.path.join(dir, alldir[i])
        filename = os.listdir(pathname)
        for j in filename:
            filename_with_dir = os.path.join(alldir[i],j)
            allFiles.append(filename_with_dir)

    validation_file = open(os.path.join(dir,"validation_list.txt"), "r")
    validation_list = validation_file.readlines()
    validation_file.close()
    # stripping \n from each element of validation list
    validation_list = map(lambda s: s.strip(), validation_list)
    print(validation_list)

    test_file = open(os.path.join(dir,"testing_list.txt"), "r")
    testing_list = test_file.readlines()
    test_file.close()
    # stripping \n from each element of testing list
    testing_list = map(lambda s: s.strip(), testing_list)

    for item in validation_list:
        if item in allFiles:
            allFiles.remove(item)

    for item in testing_list:
        if item in allFiles:
            allFiles.remove(item)

    for items in validation_list:
        newname = items.split('/')[0] + '_' + items.split('/')[-1]
        file1 = os.path.join(os.path.join(dir,items.split('/')[0]),items.split('/')[-1])
        file2 = os.path.join(validation_dirpath, newname)
        print(file2)
        shutil.copy(file1,file2)
    print('Done with writing wav files in validation folder')

    for items in testing_list:
        newname = items.split('/')[0] + '_' + items.split('/')[-1]
        file1 = os.path.join(os.path.join(dir,items.split('/')[0]),items.split('/')[-1])
        file2 = os.path.join(test_dirpath, newname)
        print(file2)
        shutil.copy(file1,file2)
    print('Done with writing wav files in test folder')

    for items in allFiles:
        temp = items.split('/')
        # sub_dir = temp[0]
        # file = temp[-1]
        newname = items.split('/')[0] + '_' + items.split('/')[-1]
        # file1 = os.path.join(os.path.join(dir,sub_dir),file)
        file1 = os.path.join(os.path.join(dir, items.split('/')[0]), items.split('/')[-1])

        # print(file1)
        file2 = os.path.join(train_dirpath, newname)
        # print(file2)
        shutil.copy(file1,file2)
        # print(file_path)
    print('Done with writing wav files in train folder')
    # DONE CREATING TRAIN/VALIDATION/TEST SETS


if __name__ == '__main__':
    main()
