# -*- coding: utf-8 -*-

import os, sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import codecs
import csv

import warnings
warnings.filterwarnings('ignore')

import time

# 定义切割图像类
class DataSeg():
	def __init__(self):
		self.data_path = 'L11-ImageColorRecog/data/images'
		self.image_names = os.listdir(self.data_path)
		self.location_data_path = 'L11-ImageColorRecog/data/seglocation_images'
        
        if not os.path.isdir(self.location_data_path):
            os.makedirs(self.location_data_path)

        self.seg_data_path = 'L11-ImageColorRecog/data/seg_images'

        if not os.path.isdir(self.seg_data_path):
            os.makedirs(self.seg_data_path)

    def seg(self):
        '''

            定义图像切割方法

        '''		
        print('start!...')

        for image_name in self.image_names:
        	img = Image.open(self.data_path + '/' + image_name)
        	loc_img = Image.open(self.data_path + '/' + image_name)
        	weight, height = img.size

        	left = (weight / 2) - 50
            right = (weight / 2) + 51
            upper = (height / 2) - 50
            lower = (height / 2) + 51

            draw = ImageDraw.Draw(loc_name)
            draw.line([(left, upper), (left, lower)], fill=(255, 0, 0), width=5)
            draw.line([(left, upper), (right, upper)], fill=(255, 0, 0), width=5)
            draw.line([(right, lower), (left, lower)], fill=(255, 0, 0), width=5)
            draw.line([(right, lower), (right, upper)], fill=(255, 0, 0), width=5)

            loc_img.save(self.location_data_path + '/' + image_name)

            box = [left, upper, right, lower]
            seg_img = img.crop(box)

            seg_img.save(self.seg_data_path + '/' + image_name)

        print('end!')

# 定义图像颜色矩特征类
class DataMoment(object):
	def __init__(self):
		# 获取全部切割图像名称
		self.seg_data_path = 'L11-ImageColorRecog/data/seg_images'
		self.segimage_names = os.listdir(self.seg_data_path)
		self.segimage_names.sort()

	def moment(self):
		'''

			定义图像颜色矩特征提取方法

		'''
		color_features = []    # 存储图像特征的特征列表
		str = ['类别', '序号', 'R通道一阶矩', 'G通道一阶矩', 'B通道一阶矩', 'R通道二阶矩',
		 'G通道二阶矩', 'B通道二阶矩', 'R通道三阶矩', 'G通道三阶矩', 'B通道三阶矩']

		color_features.append(str)

		for image_name in self.segimage_names:
			color_features = []
			image_name_list = image_name.split('.')[0].split('_')
			color_features.extend(image_name_list)
			img = Image.open(self.seg_data_path + '/' + 'image_name')
			# RGB图像得颜色分离
			r, g, b = img.split()
			r = np.array(r) / 255.0
			g = np.array(g) / 255.0
			b = np.array(b) / 255.0

			# 抽取颜色矩特征
			# 一阶颜色矩，均值


			# 二阶颜色矩，均方差


			# 三阶颜色矩，三阶中心距的立方根


def main():
	pass




if __name__ == '__main__':
	start = time.time()
    
    data = DataSeg()
    data.seg()

    end = time.time()
    print('用时 %4.2f 秒' % (end - start))
