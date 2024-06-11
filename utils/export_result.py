# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 23:09:47 2022

@author: mpastori
"""
import numpy as np
import json
import pathlib
import os
import datetime
import time
import struct
from math import isnan
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def set_output_location(experiment_name, run_path):
    #output_path = 'output/' + str(experiment_name)
    output_path = run_path + '/' + str(experiment_name)
    # check if directory for output exist, if not creates it
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # write in file the starting time
    print('>>>Starting the experiment: ', experiment_name)
    f = open(output_path + '/result.txt', "a+")
    f.write('>>>Starting the experiment: ' + experiment_name + '\n')
    f.write('Results at time: ' + str(datetime.datetime.now()) + '\n\r')
    f.close()  # close file
    return output_path


    
def export_results(outputs_list, labels_list, run_path, experiment_name,
                   confusionMat=None,
                   prodAccuracy=None,
                   averageAccuracy=None,
                   kappaCoeff=None,
                   title=''):
    """ Export classification results given images of result and ground-truth
    """
    C = len(np.unique(labels_list[0]))
    R = len(labels_list)
    confusionMatrixList = []
    cohenKappaScoreList = []
    producerAccuracyList = []
    userAccuracyList = []
    labels = np.arange(C)  # [1, 2, ... , C]

    output_path = run_path + '/' + str(str(experiment_name) + '/result.txt')

    # get accuracies and confusion matrices
    accuracy = []  # list of accuracy for each resolution
    countSuccess = 0
    countTestPixel = 0

    for r in range(R):
        width = labels_list[r].shape[1]
        height = labels_list[r].shape[0]
        # label choosed is argmax on xs
        for h in range(height):
            for w in range(width):
                # do computation for getting accuracy
                if labels_list[r][h][w] != 6:
                    countTestPixel += 1
                    if labels_list[r][h][w] == outputs_list[r][h][w]:
                        countSuccess += 1
        accuracy.append(countSuccess / countTestPixel)
        # reset sizes
        height = int(height / 2)
        width = int(width / 2)
        # reset counters
        countSuccess = 0
        countTestPixel = 0

        y_true = labels_list[r].ravel()
        y_pred = outputs_list[r].ravel()
        if confusionMat is not None:
            confusionMatrixList.append(confusion_matrix(y_true, y_pred))
        if kappaCoeff is not None:
            cohenKappaScoreList.append(cohen_kappa_score(y_true, y_pred))

    if prodAccuracy is not None:
        # compute producers accuracies
        for r in range(R):
            singleResProducerAccuracies = []
            singleResUserAccuracies = []

            for c in range(C):
                # print(confusionMatrixList[r][c][c])
                singleResProducerAccuracies.append(confusionMatrixList[r][c][c])
                singleResUserAccuracies.append(confusionMatrixList[r][c][c])

            for c1 in range(C):
                countP = 0
                countU = 0
                for c2 in range(C):
                    countP += confusionMatrixList[r][c1][c2] 
                    countU += confusionMatrixList[r][c2][c1]
                    # count += confusionMatrixList[r][c2][c1] #for user accuracies
                singleResProducerAccuracies[c1] /= countP
                singleResUserAccuracies[c1] /= countU
                
            userAccuracyList.append(singleResUserAccuracies)
            producerAccuracyList.append(singleResProducerAccuracies)

    if averageAccuracy is not None:
        averageAccuracy = []
        averageUserAccuracy = []
        for r in range(R):
            _sum = 0
            _sumUser = 0
            for c in range(C):
                if producerAccuracyList[r][c] is not np.nan:                    
                    _sum += producerAccuracyList[r][c]
                if userAccuracyList[r][c] is not np.nan:                    
                    _sumUser += userAccuracyList[r][c]
            _sum /= C
            _sumUser /= C
            averageAccuracy.append(_sum)
            averageUserAccuracy.append(_sumUser)

    # write accuracies in file
    f = open(output_path, "a+")
    f.write(title + '\n\r')
    f.write('overall accuracy\n\r')
    for r in reversed(range(R)):
        print('overall accuracy in r = ', r, ' -> ', accuracy[r])
        f.write('r = ' + str(r) + ' -> ' + str(accuracy[r]) + '\n')
    f.write('\n')
    # close file
    f.close()

    if prodAccuracy is not None:
        # write producer accuracies in file
        f = open(output_path, "a+")
        f.write('producer accuracies\n\r')
        for r in reversed(range(R)):
            f.write('r = ' + str(r) + '\n')
            for c in range(C):
                # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
                f.write('c = ' + str(c + 1) + ' -> ' + str(producerAccuracyList[r][c]) + '\n')
        f.write('user accuracies\n\r')
        for r in reversed(range(R)):
            f.write('r = ' + str(r) + '\n')
            for c in range(C):
                # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
                f.write('c = ' + str(c + 1) + ' -> ' + str(userAccuracyList[r][c]) + '\n')
      
        f.write('\n')
        # close file
        f.close()

    if averageAccuracy is not None:
        # write average accuracies in file
        f = open(output_path, "a+")
        f.write('average producer accuracies\n\r')
        for r in reversed(range(R)):
            print('average producer accuracy in r = ', r, ' -> ', averageAccuracy[r])
            f.write('r = ' + str(r) + ' -> ' + str(averageAccuracy[r]) + '\n')
        f.write('average user accuracies\n\r')
        for r in reversed(range(R)):
            print('average user accuracy in r = ', r, ' -> ', averageUserAccuracy[r])
            f.write('r = ' + str(r) + ' -> ' + str(averageUserAccuracy[r]) + '\n')
        f.write('\n')
        # close file
        f.close()

    if kappaCoeff is not None:
        # write cohen kappa score in file
        f = open(output_path, "a+")
        f.write('cohen kappa scores\n\r')
        for r in reversed(range(R)):
            # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
            f.write('r = ' + str(r) + ' -> ' + str(cohenKappaScoreList[r]) + '\n')
        f.write('\n')
        # close file
        f.close()

    if confusionMat is not None:
        # write confusion matrix in file
        f = open(output_path, "a+")
        f.write('confusion matrices\n\r')
        for r in reversed(range(R)):
            mat = np.matrix(confusionMatrixList[r])
            # with open('outfile.txt','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%i', delimiter='    ')
            f.write('\n')
        # close file
        f.close()

    # add a blank line at the bottom
    f = open(output_path, "a+")
    f.write('\n')
    f.close()
