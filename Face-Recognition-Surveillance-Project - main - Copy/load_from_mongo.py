from pymongo import MongoClient
import numpy as np
import ssl

# Bypassing the system-level handshake block entirely using an unverified context
client = MongoClient(
    "mongodb+srv://chakrateja:13032004%40R@frs.xpy5x82.mongodb.net/?appName=FRS",
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client["face"]
coll = db["users"]

def load_faces_from_mongo():
    encodings = []
    names = []
    for doc in coll.find():
        names.append(doc["name"])
        encodings.append(np.array(doc["encoding"]))
    print(f"[INFO] Loaded {len(encodings)} encodings from MongoDB.")
    return encodings, names
