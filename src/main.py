from pipeline.pipeline import VideoPipeline

from inpoutp.input import CameraInput
from inpoutp.output import VideoOutput

from debug.debug import dbg

def app_run():
    print("{Real-Time VD}: Starting the application...")
    dbg("app_run()")
    n=100
    w=1000
    h=1000
    fps=60

    cinput=CameraInput(0,w,h,fps)
    output=VideoOutput(w,h,fps)
    pipeline=VideoPipeline(n,w,h)
    cinput.capture()
    dbg("capture() happened, starting main loop")
    try:
        while True:
            frame = cinput.get_frame()
            if frame is None:
                break
            res= pipeline.process(frame)
            stop=output.show(res)
            if stop:
                break
    finally:
        dbg("break while cycle")
        cinput.release()
        output.release()
