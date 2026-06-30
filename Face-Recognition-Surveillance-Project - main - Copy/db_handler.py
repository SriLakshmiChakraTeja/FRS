# db_handler.py
import sqlite3
import cv2
from datetime import datetime
from email_alert import send_alert_email
import threading

# --- Connect to existing SQLite database ---
conn = sqlite3.connect("surveillance.db", check_same_thread=False)
cursor = conn.cursor()

# --- Ensure 'logs' table exists (won’t overwrite anything) ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    face_image BLOB
)
""")
conn.commit()

def _send_email_async(name, timestamp, frame_with_face):
    """Internal function to send email in a separate thread."""
    send_alert_email(name, timestamp, frame_with_face)

def log_detection(name, frame_with_face=None):
    """
    Store log into the existing 'logs' table in surveillance.db.
    Triggers email alert for Unknown faces.
    """
    try:
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        img_blob = None
        if frame_with_face is not None:
            try:
                _, buffer = cv2.imencode('.jpg', frame_with_face)
                img_blob = buffer.tobytes()
            except Exception as e:
                print(f"[ERROR] Failed to encode image for logging: {e}")

        cursor.execute(
            "INSERT INTO logs (name, date, time, face_image) VALUES (?, ?, ?, ?)",
            (name, date_str, time_str, img_blob)
        )
        conn.commit()

        print(f"[DB] {name} logged at {date_str} {time_str}")

        # Send email alert if unknown face detected
        if name == "Unknown" and frame_with_face is not None:
            email_thread = threading.Thread(
                target=_send_email_async,
                args=(name, f"{date_str} {time_str}", frame_with_face)
            )
            email_thread.start()

    except Exception as e:
        print(f"[ERROR] Failed to log detection: {e}")

def fetch_logs(limit=20):
    """
    Fetch latest logs from 'logs' table.
    Returns: list of tuples (id, name, date, time, face_image BLOB).
    """
    try:
        cursor.execute(
            "SELECT id, name, date, time, face_image FROM logs ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] Failed to fetch logs: {e}")
        return []

def close_db():
    """Safely close SQLite database connection."""
    try:
        conn.close()
        print("[DB] Connection closed.")
    except Exception as e:
        print(f"[ERROR] Failed to close DB: {e}")
