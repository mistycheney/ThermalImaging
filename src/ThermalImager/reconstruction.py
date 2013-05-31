'''
Created on May 26, 2013

@author: yuncong
'''

import os, sys
import subprocess
import numpy as np
import cv2
import re
from ThermalImager import config

def quaternion2matrix(w,x,y,z):
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    R = np.array([[1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                  [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
                  [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)]])
    return R

def read_vsfm_results(patch_filename, ply_filename,nvm_filename):
    patch_file = open(patch_filename, 'r')
    patch_file.readline()
    point_number = int(patch_file.readline())
    visible_from_cam = []
    for i in range(point_number):
        for j in range(5):
            patch_file.readline()
        visible_from_cam.append(map(int, patch_file.readline().split()))
        for j in range(3):
            patch_file.readline()

    nvm_file = open(nvm_filename, 'r')
    for i in range(2):
        nvm_file.readline()
    camera_num = int(nvm_file.readline())

    cameras = []
    for i in range(camera_num):
        strs = nvm_file.readline().split()
        vis_name = strs[0]
        f,w,x,y,z,camx,camy,camz,distort,zero = map(float, strs[1:])
        matches = re.search('(left|right|top|bottom|thermal)(.*)\.jpg', vis_name.split('/')[-1]).groups()
        top_or_bot = matches[0]
        visible_points = [p for p in range(point_number) if i in visible_from_cam[p]]
        pic_index = int(matches[1])
        R = quaternion2matrix(w,x,y,z)
        C = np.array([camx,camy,camz])
        T = -np.dot(R,C)
        cameras.append([top_or_bot, pic_index, visible_points, f, R,T,C,distort])
        
    ply_file = open(ply_filename, 'r')
    ply_file.readline()
    ply_file.readline()
    points_3d = []
    orig_rgb = []
    point_num = int(ply_file.readline().split()[2])
    for i in range(11):
        ply_file.readline()
    for i in range(point_num):
        x,y,z,nx,ny,nz,r,g,b,psz = map(float, ply_file.readline().split())
        orig_rgb.append([r,g,b])
        points_3d.append([x,y,z])
    
    patch_file.close()
    nvm_file.close()
    ply_file.close()
    
    from operator import itemgetter
    cameras_sorted = sorted(cameras, key=itemgetter(1))   # secondary key on pic_index
    cameras_sorted = sorted(cameras_sorted, key=itemgetter(0)) # primary key on top_or_bot
    if camera_num%2 == 1:
        if cameras_sorted[camera_num%2][0] == 'top':
            cameras_bot = cameras_sorted[:camera_num/2]
            cameras_top = cameras_sorted[camera_num/2:-1]
        else:
            cameras_bot = cameras_sorted[:camera_num/2]
            cameras_top = cameras_sorted[camera_num/2+1:]
    else:
            cameras_bot = cameras_sorted[:camera_num/2]
            cameras_top = cameras_sorted[camera_num/2:]
    return np.array(points_3d), np.array(orig_rgb), cameras_top, cameras_bot

def rescale_point_cloud(points, cameras_top, cameras_bot, R_rgb2rgb, T_rgb2rgb):
    bots = np.array([c[6] for c in cameras_bot])
    tops = np.array([c[6] for c in cameras_top])
    top_bot_dists = np.sum(np.abs(tops-bots)**2,axis=-1)**(1./2)
    C_rgb2rgb = -np.dot(np.linalg.inv(R_rgb2rgb),T_rgb2rgb)
    world_top_bot_dist = np.sum(C_rgb2rgb**2)**(1./2)
    print world_top_bot_dist
    scale = world_top_bot_dist/np.mean(top_bot_dists)
    print 'scale', scale
    points = points * scale
    for cam_top, cam_bot in zip(cameras_top, cameras_bot):
        cam_top[5] = cam_top[5]*scale
        cam_top[6] = cam_top[6]*scale
        cam_bot[5] = cam_bot[5]*scale
        cam_bot[6] = cam_bot[6]*scale
    
    return points, cameras_top, cameras_bot

def get_thermal_cameras(cameras_top,R,T):
    thermal_cameras = []
    for top_or_bot,pic_index,visible_points,f,R1,T1,C,distort in cameras_top:
        res = cv2.composeRT(cv2.Rodrigues(R1)[0], T1, cv2.Rodrigues(R)[0], T)
        rvec3 = res[0]
        tvec3 = res[1]
        thermal_cameras.append((pic_index, rvec3, tvec3))
        R_thermal = cv2.Rodrigues(rvec3)[0]
        T_thermal = tvec3
        C_thermal = -np.dot(np.linalg.inv(R_thermal),T_thermal)
        d = C - np.squeeze(C_thermal)
        dd = np.sum(np.abs(d)**2,axis=-1)**(1./2)
    return thermal_cameras

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

#     def rigid_transform_3D(A, B):
#         assert len(A) == len(B)
#         N = A.shape[0]; # total points
#         
#         centroid_A = np.mean(A, axis=0)
#         centroid_B = np.mean(B, axis=0)
#      
#         # centre the points
#         AA = A - np.tile(centroid_A, (N, 1))
#         BB = B - np.tile(centroid_B, (N, 1))
# 
#         # dot is matrix multiplication for array
#         H = np.transpose(AA) * BB        
#         U, S, Vt = np.linalg.svd(H)
#         R = Vt.T * U.T
# 
#         # special reflection case
#         if np.linalg.det(R) < 0:
#             print "Reflection detected"
#             Vt[2,:] *= -1
#             R = Vt.T * U.T        
#         t = -R*centroid_A.T + centroid_B.T
#         
#         print t
#         return R, t

if __name__ == '__main__':
    name1 = 'top'
    name2 = 'thermal'
    calib_path = 'calib3'
    
    outfile_name = config.DATASET_PATH + '/'+calib_path+'/top_thermal_calib_info.txt'
    outfile = open(outfile_name, 'r')
    npzfile = np.load(outfile)
    print npzfile.files
    K1 = npzfile['K1']
    d1 = npzfile['d1']
    K2 = npzfile['K2']
    d2 = npzfile['d2']
    R = npzfile['R']
    T = npzfile['T']

    outfile_name = config.DATASET_PATH + '/'+calib_path+'/top_bottom_calib_info.txt'
    outfile = open(outfile_name, 'r')
    npzfile = np.load(outfile)
    print npzfile.files
    R_rgb2rgb = npzfile['R']
    T_rgb2rgb = npzfile['T']
       
    model_name = 'pipe'
    nvm_filename = config.DATASET_PATH + model_name + '/model/' + model_name+'_stereo.nvm'
    patch_filename = config.DATASET_PATH + model_name + '/model/' + model_name+'_stereo.nvm.cmvs/00/models/option-0000.patch'
    
    ply_bin_filename = config.DATASET_PATH + model_name + '/model/'+model_name + '_stereo.0.ply'
    pcd_filename = config.DATASET_PATH + model_name + '/model/'+model_name + '_stereo.0.pcd'
    ply_filename = ply_bin_filename[:-4] + '.ascii.ply'
    subprocess.call(['pcl_ply2ply','--format=ascii', ply_bin_filename, ply_filename])
    subprocess.call(['pcl_ply2pcd', ply_bin_filename, pcd_filename])
    points, orig_rgb, cameras_top, cameras_bot = read_vsfm_results(patch_filename, ply_filename, nvm_filename)
    points, cameras_top, cameras_bot = rescale_point_cloud(points, cameras_top, cameras_bot, R_rgb2rgb, T_rgb2rgb)
    thermal_cameras = get_thermal_cameras(cameras_top,R,T)
    
    os.chdir(config.DATASET_PATH + model_name)
    
    import cPickle as pickle
    pickle.dump(points, open('points.pickle','wb'))
    pickle.dump(cameras_top, open('cameras_top.pickle','wb'))
    pickle.dump(cameras_bot, open('camera_bot.pickle','wb'))
    pickle.dump(thermal_cameras, open('thermal_cameras.pickle','wb'))
    pickle.dump(orig_rgb, open('orig_rgb.pickle','wb'))
    
    