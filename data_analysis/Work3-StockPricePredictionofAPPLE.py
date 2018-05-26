# -*- coding: utf-8 -*-

import os, sys

import numpy as np
import pandas as pd
from pandas_datareader import data
import pandas_datareader as pdr

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller as ADF    # ADF平衡性检验
from statsmodels.tsa.stattools import pacf as PACF
from statsmodels.tsa.stattools import acf as ACF
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox    # 白噪声检验
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf    # acf自相关系数图 & pacf偏自相关图

import warnings
warnings.filterwarnings('ignore')

def Downloading_Stock_Data(code, start_date, end_date):
    '''

        由Yahoo Finance API下载指定股票代码的股票数据

    '''
    stock = data.get_data_yahoo(code, start=start_date, end=end_date)

    out_file = open('L09-TimeSeriesPrediction/data/%s1718.csv' % code, 'w')
    stock.to_csv(out_file)
    out_file.close()

def Loading_Stock_Data(code):
    '''

        根据股票代码，从文件中读取数据，并进行初步处理

    '''
    data = pd.read_csv('L09-TimeSeriesPrediction/data/%s1718.csv' % code, index_col='Date')
    data.index = pd.to_datetime(data.index)

    return data

def order_determination(data, D_data, k):
    '''

        根据输入的data和D_data，确定阶数p，q

    '''
    # 阶数一般不超过length / 10
    # pmax = int(len(D_data) / 10)
    # qmax = int(len(D_data) / 10)
    pmax = 3
    qmax = 3
    
    data['Open'] = data['Open'].astype(float)

    bic_matrix = []    # Bayesian Information Criterion
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(data['Open'], (p, k, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    print(bic_matrix)

    p, q = bic_matrix.stack().idxmin()
    print('BIC最小的p值和q值为：%s, %s' % (p, q))

    return p, q

def Model_Determination(data):
    '''

        根据股票数据，确定对应的ARIMA(p, k, q)模型

    '''
    p, k, q = 0, 0, 0    # 原始序列
    D_data = data
    ADF_p = ADF(D_data['Open'])[1]
    acorr_ljungbox_p = list(acorr_ljungbox(D_data['Open'], lags=1)[1])[0]

    # 通过ADF检验和白噪声检验，确定可以分析的平稳的非白噪声序列
    while (ADF_p >= 0.05) or (acorr_ljungbox_p >= 0.05):
        k += 1
        D_data = data['Open'].diff(periods=k).dropna()
        ADF_p = ADF(D_data)[1]
        acorr_ljungbox_p = list(acorr_ljungbox(D_data, lags=1)[1])[0]

    p, q = order_determination(data, D_data, k)
    return p, k, q

def Price_Prediction(model, forecastnum):
    '''

        根据模型及预测窗口大小进行模型的前向预测

    '''
    # model.forecast(forecastnum)    # alpha default by 0.05
    yHat = model.forecast(forecastnum, alpha=0.01)    # change alpha to 0.01

    return yHat

def RMSE(yArr, yHatArr):
    '''

        计算均方根误差

    '''
    return np.sqrt(((yArr - yHatArr)**2).sum() / len(yArr))

def absError(yArr, yHatArr):
    '''

        计算绝对平均误差

    '''
    return abs(yArr - yHatArr).sum() / len(yArr)

def Error_Analysis(test_data, yHat):
    '''

        误差分析

    '''
    print('均方根误差为：', RMSE(test_data, yHat[0]))
    print('绝对平均误差', absError(test_data, yHat[0]))

def Stock_Price_Prediction(code, config):
    '''

        根据指定的股票代码和config设置，进行股票价格的预测

    '''
    # 股票数据的下载与加载
    Downloading_Stock_Data(code, start_date=config['start_date'], end_date=config['end_date'])
    data = Loading_Stock_Data(code)

    # 时间序列预测模型的建立与验证
    p, k, q = Model_Determination(data)
    print('对于%s股票数据，考虑建立模型ARIMA(%d, %d, %d)' % (code, p, k ,q))

    # training_sample_num = int(len(data) * 0.8)
    # testing_sample_num = len(data) - training_sample_num
    training_sample_num = len(data) - 5
    testing_sample_num = 5
    train_data = data[:training_sample_num]['Open']
    test_data = data[training_sample_num:]['Open']
    print(train_data[-1])

    model = ARIMA(train_data, (p, k, q)).fit()
    yHat = Price_Prediction(model, forecastnum=testing_sample_num)

    Error_Analysis(test_data, yHat)
    
    # 预测5天的股价
    forecastnum = config['forecastnum']
    yHat_final = Price_Prediction(model, forecastnum=forecastnum+testing_sample_num)
    print(yHat_final[0][-forecastnum:])    # 模型预测结果


def main():
    code1 = 'AAPL'
    code2 = 'BIDU'

    config = {
        'start_date': '2017/5/26',
        'end_date': '2018/5/25',
        'forecastnum': 5
    }

    Stock_Price_Prediction(code1, config)
    Stock_Price_Prediction(code2, config)

if __name__ == '__main__':
    main()