from pipeline.pipeline import VideoPipeline

from io.input import CameraInput
from io.output import VideoOutput

from debug.debug import dbg

def app_run():
    print("{Real-Time VD}: Starting the application...")
    dbg("app_run()")
    n=30
    w=500
    h=500
    fps=30

    cinput=CameraInput(0,n,w,h,fps)
    output=VideoOutput(w,h,fps)
    pipeline=VideoPipeline(n)
    cinput.capture()
    dbg("capture() happened, starting main loop")
    try:
        while True:
            frame = cinput.get_frame()
            if frame is None:
                break
            res= pipeline.process(frame)
            output.show(res)
    finally:
        dbg("break while cycle")
        input.release()
