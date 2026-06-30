# face_utils.py
import cv2
import os
import face_recognition

def load_known_faces(path='Training_images'):
    images = []
    names = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            names.append(os.path.splitext(file)[0].upper())
    return images, names

def encode_faces(images):
    encodings = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_encodings(rgb)
        if faces:
            encodings.append(faces[0])
    return encodings

def image_to_blob(img):
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()
