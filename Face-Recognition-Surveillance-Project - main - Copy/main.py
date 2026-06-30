# main.py

import cv2
import numpy as np
from datetime import datetime
from config import *
from db_handler import log_detection, close_db
from frame_reader import FrameReader
import face_recognition
from load_from_mongo import load_faces_from_mongo

known_encodings, class_names = load_faces_from_mongo()

frame_reader = FrameReader(IP_CAM_URL)
frame_reader.start()

frame_count = 5
last_logged_time = {}

try:
    while True:
        success, img = frame_reader.read()
        if not success or img is None:
            continue

        small_img = cv2.resize(img, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            faces = face_recognition.face_locations(rgb_small_img)
            encodes = face_recognition.face_encodings(rgb_small_img, faces)
            
            # Use a set to store unique names detected in this frame
            names_in_frame = set()
            
            for encode_face, face_loc in zip(encodes, faces):
                name = "Unknown"
                confidence = 0

                if known_encodings:
                    face_dis = face_recognition.face_distance(known_encodings, encode_face)
                    min_dist = np.min(face_dis)
                    confidence = (1 - min_dist) * 100

                    if min_dist < 0.5:
                        if confidence >= 60:
                            match_index = np.argmin(face_dis)
                            name = class_names[match_index]
                        else:
                            name = "Unknown"
                    else:
                        name = "Unknown"

                names_in_frame.add(name)

                top, right, bottom, left = face_loc
                y1, x2, y2, x1 = int(top / FRAME_SCALE), int(right / FRAME_SCALE), int(bottom / FRAME_SCALE), int(left / FRAME_SCALE)
                
                # Assign color based on the final name
                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                
                label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else name
                print(f"NAME : {label}")

                try:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw filled rectangle at the top of the bounding box
                    cv2.rectangle(img, (x1, y1 - 35), (x2, y1), color, cv2.FILLED)
                    
                    # Place text inside the top rectangle
                    cv2.putText(img, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                except Exception as e:
                    print(f"[ERROR] Drawing failed: {e}")
            
            # Log unique names from the frame
            now = datetime.now()
            for name in names_in_frame:
                if name not in last_logged_time or (now - last_logged_time[name]).total_seconds() > LOG_INTERVAL:
                    log_detection(name, img) 
                    last_logged_time[name] = now

        frame_count += 1
        cv2.imshow("Surveillance Feed", img)

        if cv2.waitKey(1) in [ord('q'), 27]:
            print("[INFO] Exiting...")
            break

finally:
    frame_reader.stop()
    frame_reader.join()
    close_db()
    cv2.destroyAllWindows()