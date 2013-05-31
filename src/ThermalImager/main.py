'''
Created on Apr 5, 2013

@author: yuncong
'''

import os, sys
import numpy as np
import random 
import numpy.linalg as linalg
from ThermalImager import config, capture, calibration, reconstruction, overlay, utility
import itertools
import time
import cv2
import subprocess

if __name__ == '__main__':
#     os.chdir(config.PROJPATH)
    
#     output_folder = '/home/yuncong/Documents/dataset/new'
#     if not os.path.exists(output_folder):
#         os.mkdir(output_folder)
#     capture.capture(output_folder, capture.CAPTURE_MODE.SNAPSHOT)
    
    calib_folder = '/home/yuncong/Documents/dataset/calib3/'
    K1,d1,K2,d2,R,T = calibration.calibrate(calib_folder,'top','thermal')
    
    result_path = '/home/yuncong/Documents/dataset/hand_model'
    
    patch_file_name = result_path + '/hand_model.nvm.cmvs/00/models/option-0000.patch'
    cameras_file_name = result_path + '/hand_model.nvm.cmvs/00/cameras_v2.txt'
    points_3d, cameras, camera_sees = reconstruction.read_vsfm_results(patch_file_name, cameras_file_name)
    thermal_RTs = reconstruction.get_thermal_cameras(cameras, R, T)

#     print cameras[0]
#     print thermal_RTs[0]
    
    cam_index = 21
    vis_points, thermal_colors = overlay.assign_points_thermal_data(cam_index, thermal_RTs, 
                                       K2,d2, camera_sees, points_3d)
    
    thermal_ply_name = '/home/yuncong/Documents/dataset/thermal_3d.ply'
    thermal_pcd_name = '/home/yuncong/Documents/dataset/thermal_3d.pcd'
    utility.write_ply(thermal_ply_name, vis_points, thermal_colors)
    subprocess.call(['/home/yuncong/pcl/release/bin/pcl_ply2pcd',thermal_ply_name,thermal_pcd_name], 
                    shell=True)
        
#     im_id = 336
#     im_top = cv2.imread(config.STEREODATAPATH + 'top' + str(im_id) + '.jpg')
#     im_bottom = cv2.imread(config.STEREODATAPATH + 'bottom' + str(im_id) + '.jpg')
#     
#     sgbm = StereoMatch.SGBMTuner(config.DEFAULT_SGBM_PARAMS, 'SGBMTuner', im_top, im_bottom)
    
#    print 'raw points', xyz_valid.shape[0]
#    p = pcl.PointCloud()
#    p.from_array(xyz_valid)
#    vox = p.make_voxel_grid_filter()
#    vox.set_leaf_size(0.01,0.01,0.01)
#    pv = vox.filter()
#    downsampled_cloud = pv.to_array()
#    point_number = downsampled_cloud.shape[0]
#    print 'after voxel grid filter', point_number
#    
#    downsampled_cloud = geometry.statistical_outlier_removal(downsampled_cloud, kd=None)
#    point_number = downsampled_cloud.shape[0]
#    print 'after statistical_outlier_removal', point_number
#    
#    axis_cloud = utility.draw_axis()
#    downsampled_cloud_color = utility.paint_pointcloud(downsampled_cloud, np.array([255,255,255]))
#    utility.write_XYZRGB(utility.merge_pointclouds([axis_cloud, downsampled_cloud_color]), 'downsampled_cloud.pcd')
    
    
