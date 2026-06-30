# face_viewer_gui.py

import tkinter as tk
from tkinter import ttk
from pymongo import MongoClient
from gui_components import FaceRecognitionApp
from load_from_mongo import load_faces_from_mongo
import config

def main():
    root = tk.Tk()
    
    app_state = {
        "preview_size": (440, 440),
        "camera_running": False,
        "cap": None,
        "camera_loop_id": None,
        "current_preview_imgtk": None,
        "pending_rgb_image": None,
        "name_groups": {},
        "name_order": [],
        "current_gallery_images": [],
        "current_gallery_docs": [],
        "selected_photo_indices": set(),
        "delete_btn": None,
        "live_feed_label": None,
        "known_encodings": [],
        "known_names": [],
        "status_var": tk.StringVar(value="Ready."),
        "CAM_SOURCE": config.CAM_SOURCE,
        "EMAIL_SENDER": config.EMAIL_SENDER,
        "EMAIL_RECEIVER": config.EMAIL_RECEIVER,
        "EMAIL_PASSWORD": config.EMAIL_PASSWORD,
        "LOG_INTERVAL": config.LOG_INTERVAL,
        "PROCESS_EVERY_N_FRAMES": config.PROCESS_EVERY_N_FRAMES
    }

    # MongoDB setup
    client = MongoClient(
    "mongodb+srv://chakrateja:13032004%40R@frs.xpy5x82.mongodb.net/?appName=FRS",
    tls=True,
    tlsAllowInvalidCertificates=True
)


    app = FaceRecognitionApp(root, app_state, client, load_faces_from_mongo)
    app.root.mainloop()

if __name__ == "__main__":
    main()