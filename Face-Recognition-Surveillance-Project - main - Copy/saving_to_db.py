# saving_to_db.py – Save image to folder + MongoDB, and store encoding

import os
import cv2
import face_recognition
from pymongo import MongoClient
import gridfs

# MongoDB setup
client = MongoClient(
    "mongodb+srv://chakrateja:13032004%40R@frs.xpy5x82.mongodb.net/?appName=FRS",
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client["face"]
coll = db["users"]
fs = gridfs.GridFS(db)

# Local folder for image storage
IMAGE_DIR = "Training_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def store_face(name, image_path):
    name = name.strip().upper()

    # Load image and extract encoding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print("[ERROR] No face detected in the image.")
        return

    encoding = encodings[0]

    # Save image locally
    img_bgr = cv2.imread(image_path)
    save_path = os.path.join(IMAGE_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, img_bgr)

    # Save image to MongoDB using GridFS
    with open(image_path, "rb") as f:
        image_id = fs.put(f.read(), filename=f"{name}.jpg")

    # Save name + encoding + GridFS image reference
    data = {
        "name": name,
        "encoding": encoding.tolist(),
        "image_id": image_id
    }

    coll.insert_one(data)
    print(f"[INFO] Image saved locally and to MongoDB. Encoding stored for: {name}")
