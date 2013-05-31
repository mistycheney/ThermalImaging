import cv2

class ParamsTuner(object):
    def __init__(self, params, winname):
        self.params = params
        self.winname = winname
        cv2.namedWindow(winname, 1)
        cv2.moveWindow(winname, 0, 0)
        for k,(v,r) in params.iteritems():
            cv2.createTrackbar(k, self.winname, v, r, self.onChange)
        self.onChange(None)
        if self.do_tune:
            cv2.waitKey()
        self.clean_up()
        cv2.destroyAllWindows()
    
    def onChange(self, i):
        for k in self.params.iterkeys():
            self.params[k] = cv2.getTrackbarPos(k, self.winname), self.params[k][1]
            print k, self.params[k][0]
        print
        self.doThings()
        
    def doThings(self):
        pass
    
    def clean_up(self):
        pass