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
#批量生成目录与配置文件
import os
import  CO2_v1_2
import shutil
# import configparser
inputpath='./input/'
outputpath='./output/'
#读取CALIOP数据文件列表
CALlist=os.listdir(inputpath+'CAL')
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
shutil.copy2(inputpath+'Config_Parameters.ini',
            outputpath+'/'+'Config_Parameters.ini') 
CO2_v1_2.fusion(inputpath,outputpath)