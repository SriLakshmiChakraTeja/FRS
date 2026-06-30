# camera_handler.py

import cv2
import threading
from queue import Queue
import time

class CameraThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.cap = None
        self.queue = Queue()
        self.running = threading.Event()
        self.running.set()
        self.latest_frame = None
        self.lock = threading.Lock()
        
    def run(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            print(f"Failed to open camera source: {self.src}")
            self.running.clear()
            return
            
        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.latest_frame = frame
            time.sleep(0.01) # A small sleep to prevent busy-waiting

        self.cap.release()
        print("Camera thread stopped.")

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        return False, None

    def stop(self):
        self.running.clear()
