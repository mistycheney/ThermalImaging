'''
Created on May 7, 2013

@author: yuncong
'''

from ThermalImager import config
import cv2
import numpy as np

def find_blob(im):    
    im[:30, :] = 0
    im[-30:, :] = 0
    im[:, :10] = 0
    im[:, -10:] = 0
    
#    thresh, im_bw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#    print thresh
    thresh, im_bw = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(im_bw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#    moments = [cv2.moments(i).nu20 for i in contours]
#    all_bbox = [cv2.boundingRect(i) for i in contours]
    all_area = np.array([cv2.contourArea(c) for c in contours])
#    all_aspect_ratio = np.array([float(b[2]) / b[3] for b in all_bbox])    
    print all_area
    
    big_indices = np.nonzero(all_area > config.AREA_THRESH)[0]
    print big_indices
    
    ellipses = [cv2.fitEllipse(contours[i]) for i in big_indices]
    color = [0, 255, 0]
    im_rgb = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2RGB)
    for ell in ellipses:
        center, dimension, angle = ell
        h, w = dimension
        print h, w
        cv2.ellipse(im_rgb, ell, color, 2, 8)
        
    cv2.namedWindow('threshold')
    cv2.moveWindow('threshold', 800, 100)
    cv2.imshow('threshold', im_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()
