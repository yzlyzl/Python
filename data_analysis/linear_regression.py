# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

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
    yMat = np.mat(yArr).T    # yArr is a row vector
    xTx = xMat.T * xMat    # the matrix of (X^T * X)
    if np.linalg.det(xTx) == 0.0:
        print ('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)    # the vector of Regression Coefficient
    return ws

def yPrediction(xArr, ws):
    '''

        根据计算得到的回归系数ws，来预测y

    '''
    xMat = np.mat(xArr)
    yHat = xMat * ws
    return yHat

def dataPlot(xArr, yArr):
    '''

        绘制原始数据集的散点图


    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

def main():
    data_file = 'L08-Regression/ex0.txt'
    xArr, yArr = loadDataSet(data_file)
    ws = standRegres(xArr, yArr)

    yHat = yPrediction(xArr, ws)
    dataPlot(xArr, yArr)

if __name__ == '__main__':
    main()
