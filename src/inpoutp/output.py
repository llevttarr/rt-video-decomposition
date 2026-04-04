
from debug.debug import dbg

import cv2

class VideoOutput:
    def __init__(self,w,h,fps,name="RTVD"):
        self.w=w
        self.h=h
        self.fps=fps
        self.name=name

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name,self.w, self.h)
    def show(self,res):
        if res is None:
            dbg("show(): res==None ")
            return True
        cv2.imshow(self.name,res)
        delay = max(1, int(1000/self.fps))
        key = cv2.waitKey(delay)&0xFF
        if key==ord('w'):
            dbg("show(): w pressed")
            return True
        return False
    def release(self):
        dbg("release()")
        cv2.destroyAllWindows()
