'''
Created on May 26, 2013

@author: yuncong
'''

import cv2
import numpy as np
from ThermalImager import config, utility

def assign_points_thermal_data(dataset_name, cam_index, thermal_cameras, K_thermal,d_thermal, cameras_top, points_3d):
    pic_index, rvec, tvec = thermal_cameras[cam_index]
    visible_point_indices = np.array(cameras_top[cam_index][2])
    print visible_point_indices.shape
    visible_points = np.reshape(points_3d[visible_point_indices,:3],(-1,1,3))
#     print visible_point_indices
    img_points, _ = cv2.projectPoints(visible_points, rvec, tvec, K_thermal, d_thermal)
    img_points = np.squeeze(img_points)
    img_points = np.array([[int(u),int(v)] for u,v in img_points])
    
#     print img_points

    valid = np.nonzero(np.array([u>=0 and v>=0 and u<640 and v<480 for u,v in img_points]))[0].astype(np.int)
    print valid.shape
    thermal_img = cv2.imread(config.DATASET_PATH + dataset_name +
                              '/data/thermal'+str(pic_index)+'.jpg')
    
#     thermal_img_vis = thermal_img[:]
#     for u,v, in img_points:
#         thermal_img_vis[v,u] = [255,0,0]
#     cv2.imshow(str(pic_index),thermal_img_vis)
#     cv2.waitKey()

    points_thermal = np.array([thermal_img[v,u] for u,v in img_points if u>=0 and v>=0 and u<640 and v<480])
#     points_thermal = np.array([[255, 255, 255] for u,v in img_points if u>=0 and v>=0 and u<640 and v<480])    
    return (visible_point_indices[valid], points_thermal)

if __name__ == '__main__':
    
    model_name = 'pipe'
    import os
    os.chdir(config.DATASET_PATH + model_name)
    import cPickle as pickle
    points = pickle.load(open('points.pickle','rb'))
    cameras_top = pickle.load(open('cameras_top.pickle','rb'))
    thermal_cameras = pickle.load(open('thermal_cameras.pickle','rb'))
    orig_rgb = pickle.load(open('orig_rgb.pickle','rb'))
    
    name1 = 'top'
    name2 = 'thermal'
    calib_path = 'calib3'
    outfile_name = '%s/%s/%s_%s_calib_info.txt' %(config.DATASET_PATH,calib_path,name1,name2)
    npzfile = np.load(open(outfile_name, 'r'))
    print npzfile.files
    K1 = npzfile['K1']
    d1 = npzfile['d1']
    K2 = npzfile['K2']
    d2 = npzfile['d2']
    R = npzfile['R']
    T = npzfile['T']
    print K2
    print R
    print T
    
    useful_cameras = np.nonzero([len(c[2]) for c in cameras_top])[0]
    print useful_cameras
    point_num = points.shape[0]
    points_thermal = np.zeros((point_num,))
    counter = np.zeros((point_num,))
    for cam_index in useful_cameras:
        print cam_index
        if not os.path.exists(config.DATASET_PATH + model_name +
                              '/data/thermal'+str(cam_index+1)+'.jpg'):
            continue
#         try:
        visible_point_indices, thermal_colors = assign_points_thermal_data(model_name, cam_index, thermal_cameras, 
                        K2,d2, cameras_top, points)
#         print visible_point_indices
#         print points_thermal[visible_point_indices].shape
#         print thermal_colors[:,0].shape
        a = np.column_stack((points_thermal[visible_point_indices], thermal_colors[:,0]))
        points_thermal[visible_point_indices] = np.sum(a, axis=1)
        counter[visible_point_indices] += 1
#         except TypeError as e:
#             print e
    points_thermal = [0 if counter[p] == 0 else int(round(points_thermal[p]/counter[p]))
                                   for p in range(point_num)]
    thermal_ply_name = config.DATASET_PATH + model_name+'/model/thermal_3d_stereo.ply'
    thermal_pcd_name = config.DATASET_PATH + model_name+'/model/thermal_3d_stereo.pcd'
    scaled_thermal_ply_name = config.DATASET_PATH + model_name+'/model/%s_stereo_rescaled.0.ply'%model_name
    scaled_thermal_pcd_name = scaled_thermal_ply_name[:-4]+'.pcd'
    utility.write_ply(thermal_ply_name, points, np.repeat(np.atleast_2d(points_thermal).T, 3, axis=1))
    utility.write_ply(scaled_thermal_ply_name, points, orig_rgb)
    import subprocess
    subprocess.call(['pcl_ply2pcd',thermal_ply_name,thermal_pcd_name])
    subprocess.call(['pcl_ply2pcd',scaled_thermal_ply_name,scaled_thermal_pcd_name])
    subprocess.call('pcl_viewer -multiview 1 ' + config.DATASET_PATH + model_name+'/model/thermal_3d_stereo.pcd ' 
                    + config.DATASET_PATH + model_name+'/model/%s_stereo_rescaled.0.pcd'%model_name,
                    shell=True)
#     os.system(' '.join(['pcl_viewer','-multiview', '1', config.DATASET_PATH + model_name+'/model/*.pcd']))

