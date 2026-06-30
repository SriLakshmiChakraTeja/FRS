# gui_components.py

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime, timedelta

from app_logic import AppLogic
from camera_handler import CameraThread
import config

class FaceRecognitionApp:
    def __init__(self, root, app_state, mongo_client, load_faces_func):
        self.root = root
        self.app_state = app_state
        self.mongo_client = mongo_client
        self.load_faces_func = load_faces_func
        
        self.app_logic = AppLogic(self.root, self.app_state, self.mongo_client)
        
        self.root.title("Face DB Manager — MongoDB")
        self.root.geometry("1100x700")
        self.fullscreen_enabled = False

        self.setup_styles()
        self.create_main_layout()

        self.app_logic.refresh_faces(self.face_list)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#333333")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12), foreground="#666666")
        style.configure("TButton", padding=(10, 5), font=("Segoe UI", 10, "bold"), foreground="white", background="#4a90e2", borderwidth=0)
        style.map("TButton", background=[("active", "#3b73c4")])
        style.configure("Action.TButton", background="#4a90e2", foreground="white")
        style.map("Action.TButton", background=[("active", "#3b73c4")])
        style.configure("Control.TFrame", background="#f0f0f0", padding=(0, 10, 0, 0))
        style.configure("Listbox.TFrame", background="#ffffff", relief="flat", borderwidth=1)
        style.configure("Gallery.TFrame", background="#ffffff")
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#333333", relief="solid", borderwidth=1)
        style.configure("TScrollbar", troughcolor="#f0f0f0", background="#cccccc", gripcolor="#999999")
        style.map("TScrollbar", background=[("active", "#999999")])
        style.configure("Selected.Gallery.TFrame", background="#d4e1f4", relief="solid", borderwidth=3)
        style.configure("Unselected.Gallery.TFrame", background="#ffffff", relief="solid", borderwidth=1)

    def create_main_layout(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.face_manager_tab = FaceManagerTab(self.notebook, self.app_state, self.app_logic)
        self.live_feed_tab = LiveFeedTab(self.notebook, self.app_state, self.app_logic, self.load_faces_func, self)
        self.settings_tab = SettingsTab(self.notebook, self.app_state, self.app_logic)
        
        self.notebook.add(self.face_manager_tab, text="Face Manager")
        self.notebook.add(self.live_feed_tab, text="Live Feed")
        self.notebook.add(self.settings_tab, text="Settings")

        self.status_var = self.app_state["status_var"]
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(5, 2), font=("Segoe UI", 9), background="#e0e0e0")
        self.status_bar.pack(side="bottom", fill="x")

        self.face_list = self.face_manager_tab.face_list
        self.face_list.bind("<Double-1>", self.face_manager_tab.on_list_double_click)

    def on_tab_change(self, event):
        selected_tab = event.widget.tab(event.widget.select(), "text")
        if selected_tab == "Live Feed":
            self.live_feed_tab.start_live_feed()
        else:
            self.live_feed_tab.stop_live_feed()

    def on_close(self):
        self.live_feed_tab.stop_live_feed()
        self.root.destroy()

    def toggle_fullscreen(self, event=None):
        if self.notebook.tab(self.notebook.select(), "text") == "Live Feed":
            self.fullscreen_enabled = not self.fullscreen_enabled
            self.root.attributes("-fullscreen", self.fullscreen_enabled)
            
            if self.fullscreen_enabled:
                self.status_bar.pack_forget()
                self.main_frame.pack_forget()
                self.live_feed_tab.pack(fill="both", expand=True)
            else:
                self.live_feed_tab.pack_forget()
                self.main_frame.pack(fill="both", expand=True)
                self.status_bar.pack(side="bottom", fill="x")
        
    def exit_fullscreen(self, event=None):
        if self.fullscreen_enabled:
            self.toggle_fullscreen()


class FaceManagerTab(ttk.Frame):
    def __init__(self, parent, app_state, app_logic):
        super().__init__(parent)
        self.app_state = app_state
        self.app_logic = app_logic
        
        self.app_logic.root = parent.master.master
        
        self.create_widgets()

    def create_widgets(self):
        left_panel = ttk.Frame(self, width=350, style="Listbox.TFrame")
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        self.right_panel = ttk.Frame(self)
        self.right_panel.pack(side="right", fill="both", expand=True)
        self.app_state["right_panel"] = self.right_panel

        ttk.Label(left_panel, text="Faces", style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 4))
        listbox_frame = ttk.Frame(left_panel)
        listbox_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.face_list = tk.Listbox(listbox_frame, height=26, relief="flat", borderwidth=1, highlightthickness=0, bg="#ffffff", fg="#333333", font=("Segoe UI", 10), selectbackground="#4a90e2", selectforeground="white")
        self.face_list.pack(side="left", fill="both", expand=True)
        lb_scroll = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.face_list.yview)
        lb_scroll.pack(side="right", fill="y")
        self.face_list.configure(yscrollcommand=lb_scroll.set)
        self.app_state["face_list"] = self.face_list

        toolbar = ttk.Frame(left_panel, style="Control.TFrame")
        toolbar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(toolbar, text="Add from File", command=self.app_logic.begin_add_from_file).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Add from Camera", command=self.app_logic.begin_add_from_camera).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Delete Selected", command=self.app_logic.delete_selected).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Sync", command=self.app_logic.sync_db_and_folder).pack(side="left", padx=5)

        self.app_logic.build_preview_panel(self.right_panel, "idle")

    def on_list_double_click(self, _e=None):
        self.app_logic.on_list_double_click(self.face_list, self.right_panel)


class LiveFeedTab(ttk.Frame):
    def __init__(self, parent, app_state, app_logic, load_faces_func, main_app):
        super().__init__(parent)
        self.app_state = app_state
        self.app_logic = app_logic
        self.load_faces_func = load_faces_func
        self.main_app = main_app
        self.camera_thread = None
        self.known_encodings = []
        self.known_names = []
        self.create_widgets()
        self.is_running = False
        self.frame_count = 0
        self.displayed_detections = []

    def create_widgets(self):
        live_feed_container = ttk.Frame(self, relief="solid", borderwidth=1)
        live_feed_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.app_state["live_feed_label"] = ttk.Label(live_feed_container, text="Live feed will appear here...", anchor="center", justify="center")
        self.app_state["live_feed_label"].pack(fill="both", expand=True)

        live_feed_controls = ttk.Frame(self, padding=10)
        live_feed_controls.pack(side="bottom", fill="x")
        self.start_btn = ttk.Button(live_feed_controls, text="Start Feed", command=self.start_live_feed)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(live_feed_controls, text="Stop Feed", command=self.stop_live_feed, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        ttk.Button(live_feed_controls, text="Fullscreen (F11)", command=self.main_app.toggle_fullscreen).pack(side="right", padx=5)

    def start_live_feed(self):
        if not self.is_running:
            self.known_encodings, self.known_names = self.load_faces_func()
            self.camera_thread = CameraThread(src=self.app_state["CAM_SOURCE"])
            self.camera_thread.start()
            
            self.app_logic.status("Starting live feed...")
            self.app_state["live_feed_label"].configure(text="Starting camera...")

            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="!disabled")
            
            self.after(10, self.process_live_feed)
        else:
            self.app_logic.status("Live feed is already running.")

    def stop_live_feed(self):
        if self.is_running:
            self.is_running = False
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.stop()
                self.camera_thread.join()
            self.app_logic.status("Live feed stopped.")
            
            if self.app_state["live_feed_label"]:
                self.app_state["live_feed_label"].configure(image=None, text="Live feed stopped.")
                self.app_state["live_feed_label"].image = None

            self.start_btn.configure(state="!disabled")
            self.stop_btn.configure(state="disabled")

    def process_live_feed(self):
        if not self.is_running:
            return

        ret, frame = self.camera_thread.get_frame()
        if ret:
            frame = cv2.flip(frame, 1)
            
            frame_to_process = frame.copy()

            if self.frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                small_img = cv2.resize(frame_to_process, (0, 0), fx=config.FRAME_SCALE, fy=config.FRAME_SCALE)
                rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_img)
                face_encodings = face_recognition.face_encodings(rgb_small_img, face_locations)

                current_detections = []
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"
                    confidence = 0.0
                    if self.known_encodings:
                        face_dis = face_recognition.face_distance(self.known_encodings, face_encoding)
                        match_index = np.argmin(face_dis)
                        min_dist = face_dis[match_index]
                        
                        if min_dist < 0.6:
                            confidence = (1 - min_dist) * 100
                            if confidence >= 60:
                                name = self.known_names[match_index]
                            else:
                                name = "Unknown"

                    y1, x2, y2, x1 = int(top / config.FRAME_SCALE), int(right / config.FRAME_SCALE), int(bottom / config.FRAME_SCALE), int(left / config.FRAME_SCALE)
                    current_detections.append({'name': name, 'coords': (x1, y1, x2, y2), 'confidence': confidence})
                
                self.displayed_detections = current_detections
                # self.app_logic.log_detections_from_feed(self.displayed_detections, frame)
            
            self.frame_count += 1
            
            for det in self.displayed_detections:
                x1, y1, x2, y2 = det['coords']
                name = det['name']
                confidence = det['confidence']
                
                color = (0, 255, 0)
                if name == "Unknown" or name == "Unauthorized":
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{name} ({confidence:.1f}%)" if name not in ["Unknown", "Unauthorized"] else name
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            pil_img.thumbnail((self.main_app.root.winfo_width(), self.main_app.root.winfo_height()), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=pil_img)
            
            if self.app_state["live_feed_label"]:
                self.app_state["live_feed_label"].imgtk = imgtk
                self.app_state["live_feed_label"].configure(image=imgtk, text="")
        else:
            self.app_state["live_feed_label"].configure(image=None, text="Stream is offline.")
            self.app_state["live_feed_label"].image = None

        self.after(10, self.process_live_feed)

class SettingsTab(ttk.Frame):
    def __init__(self, parent, app_state, app_logic):
        super().__init__(parent)
        self.app_state = app_state
        self.app_logic = app_logic
        self.create_widgets()
        self.load_settings()

    def create_widgets(self):
        self.settings_form = ttk.Frame(self, padding="20")
        self.settings_form.pack(expand=False, fill="both")
        
        ttk.Label(self.settings_form, text="IP Camera URL:").grid(row=0, column=0, sticky="w", pady=5, padx=5)
        self.ip_cam_entry = ttk.Entry(self.settings_form, width=50)
        self.ip_cam_entry.grid(row=0, column=1, pady=5, padx=5)

        ttk.Label(self.settings_form, text="Sender Email:").grid(row=1, column=0, sticky="w", pady=5, padx=5)
        self.sender_entry = ttk.Entry(self.settings_form, width=50)
        self.sender_entry.grid(row=1, column=1, pady=5, padx=5)

        ttk.Label(self.settings_form, text="Receiver Email:").grid(row=2, column=0, sticky="w", pady=5, padx=5)
        self.receiver_entry = ttk.Entry(self.settings_form, width=50)
        self.receiver_entry.grid(row=2, column=1, pady=5, padx=5)

        ttk.Label(self.settings_form, text="Email App Password:").grid(row=3, column=0, sticky="w", pady=5, padx=5)
        self.password_entry = ttk.Entry(self.settings_form, width=50, show="*")
        self.password_entry.grid(row=3, column=1, pady=5, padx=5)

        ttk.Label(self.settings_form, text="Log Interval (seconds):").grid(row=4, column=0, sticky="w", pady=5, padx=5)
        self.log_interval_entry = ttk.Entry(self.settings_form, width=10)
        self.log_interval_entry.grid(row=4, column=1, sticky="w", pady=5, padx=5)

        ttk.Button(self.settings_form, text="Save Settings", command=self.save_settings).grid(row=5, column=1, sticky="e", pady=10)

    def load_settings(self):
        self.ip_cam_entry.delete(0, tk.END)
        self.ip_cam_entry.insert(0, self.app_state["CAM_SOURCE"])
        self.sender_entry.delete(0, tk.END)
        self.sender_entry.insert(0, self.app_state["EMAIL_SENDER"])
        self.receiver_entry.delete(0, tk.END)
        self.receiver_entry.insert(0, self.app_state["EMAIL_RECEIVER"])
        self.password_entry.delete(0, tk.END)
        self.password_entry.insert(0, self.app_state["EMAIL_PASSWORD"])
        self.log_interval_entry.delete(0, tk.END)
        self.log_interval_entry.insert(0, str(self.app_state["LOG_INTERVAL"]))

    def save_settings(self):
        self.app_state["CAM_SOURCE"] = self.ip_cam_entry.get()
        self.app_state["EMAIL_SENDER"] = self.sender_entry.get()
        self.app_state["EMAIL_RECEIVER"] = self.receiver_entry.get()
        self.app_state["EMAIL_PASSWORD"] = self.password_entry.get()
        self.app_state["LOG_INTERVAL"] = int(self.log_interval_entry.get())
        
        messagebox.showinfo("Settings Saved", "Configuration settings have been updated for this session.")
        self.app_logic.status("Settings updated successfully.")