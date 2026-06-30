# config.py
IP_CAM_URL = 0
# IP_CAM_URL = 'http://10.229.94.153:8080/video'
# IP_CAM_URL = 'http://192.0.0.8:8080/video'
FRAME_SCALE = 0.5
PROCESS_EVERY_N_FRAMES = 30
LOG_INTERVAL = 5
DB_NAME = "Surveillance.db"

EMAIL_SENDER = 'chakratejar@gmail.com'
EMAIL_RECEIVER = 'chakrateja27@gmail.com'
EMAIL_SUBJECT = 'Unauthorized Person Detected'
EMAIL_SMTP_SERVER = 'smtp.gmail.com'
EMAIL_SMTP_PORT = 587
EMAIL_PASSWORD = 'hhda ewwh akdm lchb'

MONGO_DB_URI = "mongodb+srv://chakrateja:13032004@R@frs.xpy5x82.mongodb.net/?appName=FRS"
# MONGO_DB_URI = "mongodb+srv://mohannagasai:12345@abc@cluster1.wsgmmpi.mongodb.net/?appName=cluster1"
try:
    CAM_SOURCE = int(IP_CAM_URL)
except ValueError:
    CAM_SOURCE = IP_CAM_URL