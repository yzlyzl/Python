# -*- coding: utf-8 -*-

import os, sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller as ADF    # ADF平衡性检验
from statsmodels.tsa.stattools import pacf as PACF
from statsmodels.tsa.stattools import acf as ACF
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox    # 白噪声检验
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf    # acf自相关系数图 & pacf偏自相关图
from statsmodels.graphics.api import qqplot

import warnings
warnings.filterwarnings('ignore')

def random_series():
    # Create a random series
    x = np.random.rand(100)
    plt.plot(x)
    plt.show()

    print('ADF平衡性检验的结果为：', ADF(x))
    print('白噪声检验的结果为：', acorr_ljungbox(x, lags=1))
    plot_acf(x).show()
    plt.show()
    plot_pacf(x).show()
    plt.show()



#
#
#    Practice 1
#
#

def testing(data):
    '''

        进行ADF平衡性检验 & 白噪声检验

    '''
    print('原始序列的ADF平衡性检验的结果为：', ADF(data['volume']))
    print('原始序列的白噪声检验的结果为：', acorr_ljungbox(data['volume'], lags=1))

def differentiation(data):
    '''

         对data进行一次差分

    '''
    D_data = data.diff().dropna()
    D_data.columns = ['diff_volume']
    D_data.plot()
    plt.show()    # 差分序列的时序图

    return D_data

def order_determination(data, D_data):
    '''

        确定阶数p，q

    '''
    # 阶数一般不超过length / 10
    pmax = int(len(D_data) / 10)
    qmax = int(len(D_data) / 10)
    
    data['volume'] = data['volume'].astype(float)

    bic_matrix = []    # Bayesian Information Criterion
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    print(bic_matrix)

    p, q = bic_matrix.stack().idxmin()
    print('BIC最小的p值和q值为：%s, %s' % (p, q))

    return p, q

def residue_test(residue):
    '''

        观察ARIMA模型的残差是否是平均值为0且方差为常数的正态分布

    '''
    fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = plot_acf(residue.values.squeeze(), lags=35, ax=ax1)
    # plt.show()

    ax2 = fig.add_subplot(212)
    fig = plot_pacf(residue.values.squeeze(), lags=35, ax=ax2)
    plt.show()

    # 通过q-q图观察，检验残差是否符合正态分布
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    fig = qqplot(residue, line='q', ax=ax, fit=True)
    plt.show()

    # Ljung-Box Test - 基于一些列滞后阶数，判断序列总体的相关性或随机性是否存在
    r1, q1, p1 = ACF(residue.values.squeeze(), qstat=True)
    tmp = np.c_[list(range(1, 36)), r1[1:], q1, p1]
    table = pd.DataFrame(tmp, columns=['lag', 'AC', 'Q', 'Prob(>Q)'])
    print(table.set_index('lag')[:15])

    # 残差的白噪声检验
    print('残差的白噪声检验结果为：', acorr_ljungbox(residue, lags=1))

def saling_data_prediction(model, forecastnum):
    '''

        根据模型及预测窗口大小进行模型的前向预测

    '''
    model.forecast(forecastnum)    # alpha default by 0.05
    model.forecast(forecastnum, alpha=0.01)    # change alpha to 0.01

def practice1_saling_data_analysis():
    data = pd.read_csv('L09-TimeSeriesPrediction/data/arima_data.csv', index_col='date')
    data.index = pd.to_datetime(data.index)    # 将字符串索引转换为时间索引
    data.plot()
    plt.show()

    # plot_acf(data).show()
    # plt.show()
    # plot_pacf(data).show()
    # plt.show()
    
    # testing(data)    # 初次检验
    
    period = 0    # 原始序列
    D_data = data
    ADF_p = ADF(D_data)[1]
    acorr_ljungbox_p = list(acorr_ljungbox(D_data, lags=1)[1])[0]

    # 通过ADF检验和白噪声检验，确定可以分析的平稳的非白噪声序列
    while ADF_p >= 0.05 || acorr_ljungbox_p >= 0.05:
        period += 1
        D_data = D_data.diff(periods=period).dropna()
        ADF_p = ADF(D_data)[1]
        acorr_ljungbox_p = list(acorr_ljungbox(D_data, lags=1)[1])[0]

    p, q = order_determination(data, D_data)


    # D_data = differentiation(data)     # 一阶差分
    # plot_acf(D_data).show()
    # plt.show()    # 差分序列的自相关图
    # print('差分序列的PACF自相关系数为：', PACF(D_data['diff_volume']))
    # plot_pacf(D_data).show()
    # plt.show()
 
    # print('差分序列的ADF平衡性检验的结果为：', ADF(D_data['diff_volume']))
    # print('原始序列的白噪声检验的结果为：', acorr_ljungbox(D_data['diff_volume'], lags=1))
    # p, q = order_determination(data, D_data)

    # # 确定p, q后建立ARIMA模型
    # model = ARIMA(data, (p, 1, q)).fit()
    # resid = model.resid
    # residue_test(resid)    # 残差检验
    # model.summary2()    # 给出模型报告

    # # 模型预测
    # forecastnum = 5
    # saling_data_prediction(model, forecastnum)

    
    

#
#
#    Practice 2
#
#

def practice2_price_prediciton():
    pass



#
#
#    Practice 3
#
#

def practice3_car_saling():
    pass

def main():
    # random_series()
    practice1_saling_data_analysis()
    practice2_price_prediciton()
    practice3_car_saling()


if __name__ == '__main__':
    main()