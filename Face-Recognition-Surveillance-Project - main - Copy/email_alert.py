# email_alert.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
from config import *

def send_alert_email(name, timestamp, full_frame=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = EMAIL_SUBJECT

        body = f"Alert: Unauthorized person detected.\n\nName: {name}\nTime: {timestamp}"
        msg.attach(MIMEText(body, 'plain'))

        if full_frame is not None:
            # Resize the image for the email to save space
            scale_percent = 50
            width = int(full_frame.shape[1] * scale_percent / 100)
            height = int(full_frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_frame = cv2.resize(full_frame, dim, interpolation = cv2.INTER_AREA)

            _, img_encoded = cv2.imencode('.jpg', resized_frame)
            image_part = MIMEImage(img_encoded.tobytes(), name="intruder.jpg")
            msg.attach(image_part)

        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("[EMAIL] Alert with image sent.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")