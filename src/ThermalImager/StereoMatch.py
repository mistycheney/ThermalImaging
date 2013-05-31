from ThermalImager.ParamsTuner import ParamsTuner
from ThermalImager import config, utility
import cv2
import numpy as np
import random
import numpy.linalg as linalg

class SGBMTuner(ParamsTuner):
    def __init__(self, params, winname, top, bottom):
        self.top = top
        self.bottom = bottom
        
        top_small = cv2.resize(top, (top.shape[1] / 2, top.shape[0] / 2))
        bottom_small = cv2.resize(bottom, (bottom.shape[1] / 2, bottom.shape[0] / 2))
        cv2.imshow('top', top_small);
        cv2.imshow('bottom', bottom_small);

        extrinsic_filepath = config.PROJPATH + 'extrinsics.yml'
        intrinsic_filepath = config.PROJPATH + 'intrinsics.yml'
        self.R = np.asarray(cv2.cv.Load(extrinsic_filepath, name='R'))
        self.T = np.asarray(cv2.cv.Load(extrinsic_filepath, name='T'))
        self.R1 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='R1'))
        self.R2 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='R2'))
        self.P1 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='P1'))
        self.P2 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='P2'))
        self.Q = np.asarray(cv2.cv.Load(extrinsic_filepath, name='Q'))
        self.M1 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='M1'))
        self.M2 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='M2'))
        self.D1 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='D1'))
        self.D2 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='D2'))
        
        self.do_tune = config.TUNE_DISPARITY_MAP
        
        R1, R2, P1, P2, self.Q, topValidRoi, bottomValidRoi = cv2.stereoRectify(self.M1, self.D1, self.M2, self.D2,
                        (self.top.shape[1], self.top.shape[0]), self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        top_map1, top_map2 = cv2.initUndistortRectifyMap(self.M1, self.D1, R1, P1,
                                    (self.top.shape[1], self.top.shape[0]), cv2.CV_16SC2)
        bottom_map1, bottom_map2 = cv2.initUndistortRectifyMap(self.M2, self.D2, R2, P2,
                                (self.bottom.shape[1], self.bottom.shape[0]), cv2.CV_16SC2)
        
        self.top_r = cv2.remap(self.top, top_map1, top_map2, cv2.cv.CV_INTER_LINEAR);
        self.bottom_r = cv2.remap(self.bottom, bottom_map1, bottom_map2, cv2.cv.CV_INTER_LINEAR)
        
        top_r_small = cv2.resize(self.top_r, (self.top_r.shape[1] / 2, self.top_r.shape[0] / 2))
        bottom_r_small = cv2.resize(self.bottom_r, (self.bottom_r.shape[1] / 2, self.bottom_r.shape[0] / 2))
        cv2.imshow('top rectified', top_r_small);
        cv2.imshow('bottom rectified', bottom_r_small);
        
        tx1,ty1,tx2,ty2 = topValidRoi
        bx1,by1,bx2,by2 = bottomValidRoi
        self.roi = (max(tx1, bx1), max(ty1, by1), min(tx2, bx2), min(ty2, by2))
        self.top_r = cv2.blur(self.top_r, (5, 5))
        self.bottom_r = cv2.blur(self.bottom_r, (5, 5))
#        top_r = cv2.equalizeHist(self.top_r)
#        bottom_r = cv2.equalizeHist(self.bottom_r)
        
        super(SGBMTuner, self).__init__(params, winname)
    
    def doThings(self):
        sgbm = cv2.StereoSGBM()
        sgbm.SADWindowSize, maxDisp, sgbm.preFilterCap, sgbm.minDisparity, \
        sgbm.uniquenessRatio, sgbm.speckleWindowSize, sgbm.P1, sgbm.P2, \
        sgbm.speckleRange = [v for v, _ in self.params.itervalues()]
        sgbm.numberOfDisparities = maxDisp - sgbm.minDisparity
        sgbm.disp12MaxDiff = 1
        sgbm.fullDP = True

        self.disp = sgbm.compute(np.rot90(self.top_r,1), 
                    np.rot90(self.bottom_r,1)).astype(np.float32) / 16.0
        self.disp = np.rot90(self.disp,3)
        disp8 = (self.disp - sgbm.minDisparity) / sgbm.numberOfDisparities
        self.roi_mask = np.zeros_like(self.disp).astype(np.bool)
        self.roi_mask[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]] = True
#        dispRgb = cv2.cvtColor(disp8, cv2.COLOR_GRAY2RGB)
        dispRgb = utility.get_heatmap(disp8)
        self.mask = self.disp > self.disp.min()
        dispRgb[~(self.mask*self.roi_mask)] = [255,255,255]
        disp_small = cv2.resize(dispRgb, (dispRgb.shape[1] / 2, dispRgb.shape[0] / 2))
        cv2.imshow(self.winname, disp_small)
        
    def clean_up(self):
        self.xyz = cv2.reprojectImageTo3D(self.disp, self.Q, handleMissingValues=True)
        self.xyz_mask = np.all(~np.isinf(self.xyz),2)        
        out_points = self.xyz[self.mask * self.roi_mask * self.xyz_mask]
#        self.top_r_col = cv2.cvtColor(self.top_r, cv2.COLOR_GRAY2RGB)
        self.top_r_rgb = cv2.cvtColor(self.top_r, cv2.COLOR_BGR2RGB)
        out_colors = self.top_r_rgb[self.mask * self.roi_mask * self.xyz_mask]
        nout = out_points.shape[0]
        self.xyzrgb = np.array([np.append(point, utility.color_to_float(color)) 
                       for point,color in zip(out_points, out_colors)])
              
#        import itertools
#        self.xyzrgb_valid = np.array([i for i in itertools.chain(*self.xyzrgb) if i[2] != 10000.]).astype(np.float32)
        axis_cloud = utility.draw_axis()
        utility.write_XYZRGB(utility.merge_pointclouds([axis_cloud, self.xyzrgb]), 'lobby.pcd')
        
        utility.write_ply('lobby.ply', out_points, out_colors)