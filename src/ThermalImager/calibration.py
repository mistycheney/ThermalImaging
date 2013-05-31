'''
Created on May 10, 2013

@author: yuncong
'''

import cv2
import numpy as np

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

CALIB_MODE = enum('RGB_RGB','RGB_THERMAL')

def detect_points_from_pattern(img_mask, is_thermal=False):
    square_size = 2.15
    pattern_size = (9, 6)
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    from glob import glob
    import re
    img_names = glob(img_mask)
    img_names = sorted(img_names, key=lambda name: re.search('(left|right|top|bottom|thermal)(.*)\.(jpg|JPG)', name).groups()[1])
#        debug_dir = None
    obj_points = []
    img_points = []
    valid = []
#         h, w = 0, 0
    for i, fn in enumerate(img_names):
        print 'processing %s...' % fn,
        img_rgb = cv2.imread(fn)
#             h, w = img.shape[:2]
        cv2.imshow('original',img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        if is_thermal:
            cv2.imshow('original',img)
            cv2.waitKey()
            cv2.imshow('negative', 255-img)
            cv2.waitKey()
            found, corners = cv2.findChessboardCorners(255-img, pattern_size)
#                                         flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)
#                     flags=cv2.cv.CV_CALIB_CB_NORMALIZE_IMAGE)
        else:
            found, corners = cv2.findChessboardCorners(img, pattern_size)
        
        if found:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            if is_thermal:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                vis = img_rgb
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            cv2.imshow('vis', vis)
            cv2.waitKey()
            if is_thermal:
                cv2.imwrite('/home/yuncong/Desktop/thermal_debug'+str(i)+'.png', vis)
            else:
                cv2.imwrite('/home/yuncong/Desktop/debug'+str(i)+'.png', vis)

        else:
            print 'chessboard not found'
            continue
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)
        valid.append(i)

        print 'ok'
    return img_points, obj_points, valid


def calibrate(folder, name1, name2):    
    im1 = cv2.imread(folder + '/' + name1 +'1.jpg')
    h1 = im1.shape[0]
    w1 = im1.shape[1]
    
    im2 = cv2.imread(folder + '/' + name2 + '1.jpg')
    if im2 is None:
        im2 = cv2.imread(folder + '/' + name2 + '1.JPG')
    h2 = im2.shape[0]
    w2 = im2.shape[1]
    
    img_points1, obj_points1, valid1 = detect_points_from_pattern(folder+'/'+name1+ '*.jpg')
    if name2 == 'thermal':
        img_points2, obj_points2, valid2 = detect_points_from_pattern(folder+'/'+name2+ '*.jpg', is_thermal=True)
    else:
        img_points2, obj_points2, valid2 = detect_points_from_pattern(folder+'/'+name2+ '*.jpg')
        
    print 'valid1',valid1
    print 'valid2',valid2
    rms1, K1, d1, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points1, img_points1, (w1, h1),
                                                flags=cv2.CALIB_RATIONAL_MODEL)    
    rms2, K2, d2, rvecs2, tvecs2 = cv2.calibrateCamera(obj_points2, img_points2, (w2, h2),
                                                flags=cv2.CALIB_RATIONAL_MODEL)
        
    print 'rms1:',rms1, 'rms2:',rms2
    
    obj_points = [obj_points1[j] for j,i in enumerate(valid1) if i in valid2]
    img_points1 = [img_points1[j] for j,i in enumerate(valid1) if i in valid2]
    img_points2 = [img_points2[valid2.index(i)] for j,i in enumerate(valid1) if i in valid2]
    
    # imageSize - Size of the image used only to initialize intrinsic camera matrix. Since we fix intrinsic matrix, (w,h) below can be anything.
    rms, _, _, _, _, R,T,E,F= cv2.stereoCalibrate(obj_points, img_points1, img_points2, (w1,h1), K1,d1,K2,d2, 
                                                  flags=cv2.CALIB_FIX_INTRINSIC|cv2.CALIB_RATIONAL_MODEL)
    print 'rms:',rms
    print 'R:',R
    print 'T:',T
    
    out_file_name = '/home/yuncong/Documents/dataset/calib3/'+name1+'_'+name2+'_calib_info.txt'
    out_file = open(out_file_name, 'w')
    np.savez(out_file, K1=K1,d1=d1,K2=K2,d2=d2,R=R,T=T)
    
    return K1,d1,K2,d2,R,T

def calibrate3(folder):
    calibrate(folder, 'top', 'thermal')
    calibrate(folder, 'top', 'bottom')
    

if __name__ == '__main__':
# #     thre, im = cv2.threshold(im, 130, 255, cv2.THRESH_BINARY)
# #     cv2.imshow('thres', im)
    calibrate3('/home/yuncong/Documents/dataset/calib3/')

    