# frame_reader.py
import cv2
from threading import Thread, Lock

class FrameReader(Thread):
    def __init__(self, src):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.lock = Lock()
        self.running = True

    def run(self):
        while self.running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                self.running = False

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.running = False
        self.capture.release()
