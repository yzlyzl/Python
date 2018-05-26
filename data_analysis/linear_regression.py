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
    # the number of Fields
    # notice that the 1st & 2nd columns stand for x0 and x1, and the last column stands for y
    numFeat = len(open(fileName).readline().split('\t')) - 1    
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




################################
#
#       OLS 普通最小二乘法 
#       - 使得整体的均方误差最小
#
################################

def standRegres(xArr, yArr):
    '''

        标准回归训练, 返回回归系数ws的向量

    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T    # yArr is a row vector
    xTx = xMat.T * xMat    # the matrix of (X^T * X)
    # to calculate the determinant of xTx
    if np.linalg.det(xTx) == 0.0:
        print ('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)    # the vector of Regression Coefficient
    return ws

def yPrediction(xArr, ws):
    '''

        根据计算得到的回归系数ws，来预测y，返回预测向量yHat

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
    
    # choose the 2nd column of X which is x1, and 1st column of y
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # 对于ipython notebook可以直接用%matplotlib inline自动显示图片
    # 命令行中需要使用该条命令
    plt.show()    


##################################
#
#      LWLR 局部加权线性回归 
#      - 每条测试数据都对整体发挥作用，得到一个独立的权重矩阵来进行预测，而非所有数据共用一个系数向量ws
#
##################################

def lwlr(testPoint, xArr, yArr, k=1.0):
    '''

        局部加权线性回归训练，
        返回一个预测值testPoint * ws（当然原则上来说是一个1*1的矩阵）

        testPoint是1*(n+1)的array, xArr是m*n的array, yArr是m*1的array

    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]    # 样本个数
    # 创建对角矩阵weights，是个方阵，阶数等于样本个数，为每个样本点初始化一个权重
    weights = np.mat(np.eye(m))

    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        # 根据高斯核的对应权重
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    # the vector((n+1)*1的向量) of the Regression Coefficients
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''

        为数据集中的每个点调用lwlr()，有助于求解k的大小

    '''
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)    # 创建一个一维array
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def lwlrTestPlot(xArr, yArr, k=1.0):
    yHat = np.zeros(np.shape(yArr))
    xCopy = np.mat(xArr)
    xCopy.sort(0)
    m = np.shape(xArr)[0]

    for i in range(m):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


#####################################
#
#     Example 1 预测鲍鱼的年龄
#
#####################################

def rssError(yArr,yHatArr):
    '''

        计算误差平方和（squre sum of error，SSE）

    '''
    return ((yArr - yHatArr)**2).sum()



######################################
#
#    数据特征个数 > 样本个数 - 岭回归
#
######################################

def ridgeRegres(xMat, yMat, lam=0.2):
    '''

        岭回归训练

    '''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam

    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    '''

        岭回归在一组lambda上测试的结果，
        返回一个由不同lambda值确定的多组回归系数向量ws组成的矩阵

    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据的标准化
    yMean = np.mean(yMat, 0)    # 一个1*1的矩阵，0代表按列求均值
    yMat = yMat - yMean

    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar

    # 取30个不同的lambda值，得到30组回归系数ws
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))    # 参数需要为tuple形式
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat



#######################################
#
#      前向逐步回归 - 用于替代Iasso
#
#######################################

def regularize(xMat):
    '''

        对特征按照均值为0，方差为1进行标准化处理

    '''
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''

        前向逐步线性回归，返回一个ndarray

        eps：每次迭代需要调整的步长
        numIt：迭代次数

    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据标准化
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        lowestError = np.inf    # 误差初始值设为正无穷
        for j in range(n):    # 在所有特征上运行两次for循环
            for sign in [-1, 1]:    # 分别计算增加或减少该特征对误差的影响
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                # 如果误差变小了，那么就更新使得误差最小的回归系数wsMax
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        # 每次迭代结束对于ws进行一次更新
        ws = wsMax.copy()
        returnMat[i, :] = ws.T    # 每次迭代得到一个ws行向量
    return returnMat




def main():
    data_file = 'L08-Regression/ex0.txt'
    xArr, yArr = loadDataSet(data_file)
    # ws = standRegres(xArr, yArr)

    # yHat = yPrediction(xArr, ws)
    dataPlot(xArr, yArr)    # 命令行里面图画不出来啊，虽然能正常运行，也不报错

if __name__ == '__main__':
    main()
