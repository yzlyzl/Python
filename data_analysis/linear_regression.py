# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

np.random.seed(12345)
np.set_printoptions(precision=4)
plt.rc('figure', figsize=(10, 6))
br = '\n'

# from typing import Tuple, List

def loadDataSet(fileName):
    '''

        从文件中加载数据

    '''
    numFeat = len(open(fileName).readline().split('\t')) - 1    # the number of Fields
    dataArr = []
    labelArr = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
        labelArr.append(float(curLine[-1]))    # the label tag
    return dataArr, labelArr    # dataArr is of 2 dimensions & labelArr is of 1 dimension

def standRegres(xArr, yArr):
    '''

        标准回归训练, 返回回归系数的向量

    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat    # the matrix of (X^T * X)
    if np.linalg.det(xTx) == 0.0:
        print ('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)    # the vector of Regression Coefficient
    return ws

if __name__ == '__main__':
    pass
