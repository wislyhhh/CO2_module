# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:34:05 2022

@author: Zhejiang University, Lidar Group of Prof.LiuDong

@email: liudongopt@zju.edu.cn
"""
#清除运行环境变量
import sys

sys.path = [path for path in sys.path 
            if not (path.startswith('./') or path.endswith('.egg'))]
import  CO2_v1_2
inputpath='./input/'
outputpath='./output/'
CO2_v1_2.retrieval(outputpath)