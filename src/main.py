from pipeline.pipeline import VideoPipeline

from io.input import CameraInput
from io.output import VideoOutput

from debug.debug import dbg

def app_run():
    dbg("app_run()")
    n=30
    w=500
    h=500
    fps=30

    input=CameraInput(0,n,w,h,fps)
    output=VideoOutput(w,h,fps)
    pipeline=VideoPipeline(n)
    input.capture()
    try:
        while True:
            frame = input.get_frame()
            if frame is None:
                break
            pipeline.process(frame)
    finally:
        dbg("break while cycle")
        input.release()
