#!/bin/python

from cv2 import cv
import cv2, re
import sys, os
import time
import re
import subprocess
from ThermalImager import config

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

CAPTURE_MODE = enum('SNAPSHOT','RECORD')

def jpeg2avi():
    os.chdir('/home/yuncong/Documents/dataset/will/data2')
    vw_top = cv2.VideoWriter('../body_bottom.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 20, (640,480))
#     vw_bot = cv2.VideoWriter('../clip3_bot.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 20, (640,480))
    
    fns = os.listdir('.')
    top_names = sorted([f for f in fns if f.startswith('bottom')], key=lambda name: int(re.search('bottom(.*)\.jpg',name).groups()[0]))
#     bot_names = sorted([f for f in fns if f.startswith('bottom')], key=lambda name: int(re.search('bottom(.*)\.jpg',name).groups()[0]))
    
    if vw_top.isOpened():
        for img_name in top_names:
            img = cv2.imread(img_name)
            cv2.imshow('top',img)
        #     cv2.waitKey()
            vw_top.write(img)
    
#     if vw_bot.isOpened():
#         for img_name in bot_names:
#             img = cv2.imread(img_name)
#             cv2.imshow('bot',img)
#         #     cv2.waitKey()
#             vw_bot.write(img)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

#proj_path = '/Users/yuncong/Documents/StairsModelingPy/'

def capture(output_folder, mode, name1, name2, rgb_id1, rgb_id2, ir_id, interval=100):
    os.chdir(output_folder)
    files = os.listdir('.')
#     for i in range(208,243):
#         os.system('mv top'+str(i)+'.jpg top'+str(i-134)+'.jpg')
    nexisting = len([i for i in files if name1 in i and '.jpg' in i])
    counter = nexisting + 1    
    print counter
    
    os.system("v4l2-ctl -d /dev/video"+str(rgb_id1)+" -p 15 --set-fmt-video=width=640,height=480,pixelformat=0")
    vc1 = cv2.VideoCapture(rgb_id1)
    print 'First camera initiated'
    print "Frame width: ", vc1.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    print "Frame height: ", vc1.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)    

    os.system("v4l2-ctl -d /dev/video"+str(rgb_id2)+" -p 15 --set-fmt-video=width=640,height=480,pixelformat=0")
    vc2 = cv2.VideoCapture(rgb_id2)
    print 'Second camera initiated'
    print "Frame width: ", vc2.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    print "Frame height: ", vc2.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    os.system("v4l2-ctl -d /dev/video"+str(ir_id)+" -i 1 -s 0 --set-fmt-video=width=640,height=480,pixelformat=0")

    cv2.namedWindow(name1,1)
    cv2.moveWindow(name1, 0, 0)    
    cv2.namedWindow(name2,1)
    cv2.moveWindow(name2, 500, 0)
    
    print 'Press any key to start capture...'
    begun = False
    timer = 0
    while True:
    #    begin = time.time()
        rval1 = vc1.grab()
        rval2 = vc2.grab()
        rval11, frame1 = vc1.read()
        rval22, frame2 = vc2.read()
#         frame_small = cv2.resize(frame1, (400,300))
#         frame2_small = cv2.resize(frame2, (400,300))
        cv2.imshow(name1, frame1)
#         import numpy as np
#         print frame2.shape
#         print np.nonzero(np.reshape(frame2, (1,-1)))[1]
#         t = np.reshape(frame2,(-1,3))[:110976]
#         print t
#         tt = np.reshape(frame2,(-1,3))[110976:]
#         print tt
#         frame2 = np.reshape(t,(289,384,3))
#         frame2 = np.hstack((frame2[np.arange(0,288,2)],frame2[np.arange(1,289,2)]))        
        cv2.imshow(name2, frame2)

        key = cv2.waitKey(20)
    #    print '1', time.time() - begin, 'second'
    #    print '2', time.time() - begin, 'second'
        if not begun and key != -1: 
            begun = True
            print 'RECORD mode.' if mode == CAPTURE_MODE.RECORD else 'SNAPSHOT mode. Press space to capture one frame.'
        if begun:
            if key == 27: break
            elif key in range(255) and ((mode == CAPTURE_MODE.SNAPSHOT and chr(key) == ' ') or (mode == CAPTURE_MODE.RECORD and timer > interval)):
                timer = 0
                cv2.imwrite(name1+str(counter)+".jpg",frame1)
                cv2.imwrite(name2+str(counter)+".jpg",frame2)

                subprocess.call(('mplayer -frames 1 -vo jpeg:outdir='+output_folder+' -ao null  tv:// -tv device=/dev/video'+
                                 str(ir_id)+':input=1:outfmt=rgb24:driver=v4l2:width=640:height=480:fps=10:norm=1').split())
                subprocess.call(('mv 00000001.jpg thermal%d.jpg'%counter).split())
                
                print 'image pair', counter, 'saved'
                counter = counter + 1
    #    print time.time() - begin, 'second'
            
            if mode == CAPTURE_MODE.RECORD:
                timer += 1
                if (interval-timer)%10 == 0:
                    print 'next capture in', (interval-timer)/10, '...'
#         print timer

#     if vc1.isOpened():
    vc1.release()
#     if vc2.isOpened():
    vc2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    jpeg2avi()
    sys.exit()
    
    model_name = 'body'
    output_folder = config.DATASET_PATH + '/%s/data'%model_name
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    capture(output_folder, CAPTURE_MODE.SNAPSHOT, 'top','bottom',0,1,3)
#     capture(output_folder, CAPTURE_MODE.RECORD, 'top','bottom',1,2,3,100)