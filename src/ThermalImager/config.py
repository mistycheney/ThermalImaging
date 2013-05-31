'''
Created on Apr 16, 2013

@author: yuncong
'''
import numpy as np
from collections import OrderedDict

DATASET_PATH='/home/yuncong/Documents/dataset/'
PROJPATH='/home/yuncong/workspace/ThermalImager/'

AREA_THRESH = 10000

TUNE_DISPARITY_MAP = True
DEFAULT_SGBM_PARAMS = OrderedDict([('SADWindowSize',(5,51)),
                              ('maxDisparity',(400,1000)),
                              ('preFilterCap',(0,1000)),
                              ('minDisparity',(0,1000)),
                              ('uniquenessRatio',(3,20)),
                              ('speckleWindowSize',(0,1000)),
                              ('P1',(600,100000)),
                              ('P2',(2400,100000)),
                              ('speckleRange',(1,10))])

 
