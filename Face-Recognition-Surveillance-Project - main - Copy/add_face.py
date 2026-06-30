import os
import cv2
import gridfs
from pymongo import MongoClient
from bson import ObjectId
import face_recognition
import numpy as np
from datetime import datetime

# ====== CONFIG ======
TRAINING_DIR = "Training_images"
# Use the same MongoDB URI as the main project for consistency
MONGO_DB_URI = "mongodb+srv://chakrateja:13032004%40R@frs.xpy5x82.mongodb.net/?appName=FRS"

DB_NAME = "face"
COLLECTION_NAME = "users"

# ====== CONNECT TO MONGO ======
client = MongoClient(MONGO_DB_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)
collection = db[COLLECTION_NAME]

# ====== ENSURE TRAINING DIR EXISTS ======
if not os.path.exists(TRAINING_DIR):
    os.makedirs(TRAINING_DIR)


def save_face_image(name, image_path):
    """
    Saves a new face image locally and in MongoDB.
    Handles duplicate names by creating folder structure only when needed.
    """
    name = name.strip().upper()

    # Load image and extract encoding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print(f"[ERROR] No face detected in the image: {image_path}. Skipping.")
        return

    encoding = encodings[0]

    # Decide storage path
    person_folder = os.path.join(TRAINING_DIR, name)
    loose_file_path = os.path.join(TRAINING_DIR, f"{name}.jpg")

    is_first_photo = not os.path.exists(person_folder) and not os.path.exists(loose_file_path) and collection.count_documents({"name": name}) == 0

    if is_first_photo:
        save_path = loose_file_path
        filename_in_db = f"{name}.jpg"
    else:
        os.makedirs(person_folder, exist_ok=True)
        filename_in_db = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_path = os.path.join(person_folder, filename_in_db)

    # Save image locally
    img_bgr = cv2.imread(image_path)
    cv2.imwrite(save_path, img_bgr)

    # Save to MongoDB GridFS
    with open(save_path, "rb") as f:
        file_id = fs.put(f.read(), filename=filename_in_db)

    # Save name + encoding + GridFS image reference
    # Corrected: Store encoding as a list of floats, not a pickled object
    data = {
        "name": name,
        "image_id": file_id,
        "encoding": encoding.tolist(),
        "timestamp": datetime.now()
    }

    collection.insert_one(data)
    print(f"[INFO] Saved {os.path.basename(save_path)} and synced with MongoDB")
    

def sync_with_mongo():
    """
    Ensures Training_images and MongoDB are fully in sync.
    """
    print("[INFO] Starting sync between folder and MongoDB...")

    db_entries = list(collection.find({}))
    db_filenames = {fs.get(entry["image_id"]).filename for entry in db_entries if fs.exists(entry["image_id"])}

    local_files = set()
    for root, _, files in os.walk(TRAINING_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                local_files.add(f)
    
    # Remove DB entries without a corresponding local file
    for entry in db_entries:
        try:
            grid_out = fs.get(entry["image_id"])
            if grid_out.filename not in local_files:
                fs.delete(entry["image_id"])
                collection.delete_one({"_id": entry["_id"]})
                print(f"[SYNC] Deleted DB entry for missing local file: {grid_out.filename}")
        except Exception as e:
            print(f"[SYNC ERROR] Could not process DB entry for deletion: {e}")
            
    # Remove local files without a corresponding DB entry
    for f in local_files:
        if f not in db_filenames:
            try:
                # Find the correct path for loose files vs. files in folders
                file_path = os.path.join(TRAINING_DIR, f)
                if not os.path.exists(file_path):
                    # Check in subdirectories
                    name_part = os.path.splitext(f.split("_")[0])[0]
                    file_path = os.path.join(TRAINING_DIR, name_part, f)
                    
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[SYNC] Deleted local file missing in DB: {f}")
            except Exception as e:
                print(f"[SYNC ERROR] Could not delete local file {f}: {e}")
                
    # Clean up empty directories
    for root, dirs, _ in os.walk(TRAINING_DIR, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"[SYNC] Removed empty directory: {d}")

    print("[INFO] Sync complete.")


def close_db():
    client.close()

# Removed the if __name__ == "__main__": block to prevent automatic execution on import.