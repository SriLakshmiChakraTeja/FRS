# app_logic.py

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import io
import os
import re
import cv2
import face_recognition
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
import numpy as np
from datetime import datetime
from email_alert import send_alert_email
from db_handler import log_detection

class AppLogic:
    def __init__(self, root, app_state, mongo_client):
        self.root = root
        self.app_state = app_state
        self.client = mongo_client
        self.db = self.client["face"]
        self.coll = self.db["users"]
        self.fs = gridfs.GridFS(self.db)
        self.TRAIN_DIR = "Training_images"
        self.current_detections = []
        self.last_logged_time = {}

    def status(self, msg: str):
        self.app_state["status_var"].set(msg)
        self.root.update_idletasks()
        
    def pil_from_np(self, np_img, size=None):
        img = Image.fromarray(np_img)
        if size:
            img.thumbnail(size, Image.LANCZOS)
        return img

    def pil_from_bytes(self, b: bytes, size=None):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        if size:
            img.thumbnail(size, Image.LANCZOS)
        return img

    def set_preview_from_pil(self, pil_img, title="", subtitle=""):
        imgtk = ImageTk.PhotoImage(pil_img)
        self.app_state["current_preview_imgtk"] = imgtk
        if self.app_state["preview"] is not None:
            self.app_state["preview"].configure(image=imgtk)
        self.app_state["title_var"].set(title)
        self.app_state["subtitle_var"].set(subtitle)

    def clear_preview(self):
        if self.app_state["preview"] is not None:
            self.app_state["preview"].configure(image="")
        self.app_state["current_preview_imgtk"] = None
        self.app_state["title_var"].set("No selection")
        self.app_state["subtitle_var"].set("")

    def gridfs_filename_of(self, doc):
        try:
            gf = self.fs.get(ObjectId(doc["image_id"]))
            return gf.filename or ""
        except Exception:
            return ""

    _suffix_re = re.compile(r"_(\d+)\.jpg$", re.IGNORECASE)
    def suffix_number(self, filename: str) -> int:
        m = self._suffix_re.search(filename)
        return int(m.group(1)) if m else 0

    def refresh_faces(self, face_list_widget):
        docs = list(self.coll.find())
        groups = {}
        for d in docs:
            groups.setdefault(d["name"], []).append(d)
        for name, arr in groups.items():
            arr.sort(key=lambda x: self.suffix_number(self.gridfs_filename_of(x)))
        self.app_state["name_groups"] = dict(sorted(groups.items(), key=lambda kv: kv[0]))
        self.app_state["name_order"] = list(self.app_state["name_groups"].keys())
        face_list_widget.delete(0, tk.END)
        for name in self.app_state["name_order"]:
            count = len(self.app_state["name_groups"][name])
            face_list_widget.insert(tk.END, f"{name} ({count})")
        self.status(f"Loaded {sum(len(v) for v in self.app_state['name_groups'].values())} photos across {len(self.app_state['name_groups'])} people.")

    def sync_db_and_folder(self):
        os.makedirs(self.TRAIN_DIR, exist_ok=True)
        restored = 0
        filenames_in_db = set()
        db_docs = list(self.coll.find())
        
        db_by_name = {}
        for doc in db_docs:
            try:
                gf = self.fs.get(ObjectId(doc["image_id"]))
                fname = gf.filename or ""
                if fname:
                    filenames_in_db.add(fname)
                    name_part = os.path.splitext(fname.split("_")[0])[0]
                    db_by_name.setdefault(name_part, []).append((fname, doc["image_id"]))
            except Exception as e:
                print(f"Error reading from GridFS: {e}")

        for name_part, entries in db_by_name.items():
            person_dir = os.path.join(self.TRAIN_DIR, name_part)
            
            if len(entries) > 1:
                os.makedirs(person_dir, exist_ok=True)
                
                loose_file_path = os.path.join(self.TRAIN_DIR, f"{name_part}.jpg")
                if os.path.exists(loose_file_path):
                    try:
                        new_loose_filename = f"{name_part}_1.jpg"
                        if not os.path.exists(os.path.join(person_dir, new_loose_filename)):
                            os.rename(loose_file_path, os.path.join(person_dir, new_loose_filename))
                            print(f"Moved loose file '{name_part}.jpg' to folder.")
                    except Exception as e:
                        print(f"Failed to move loose file to folder: {e}")

                for fname, img_id in entries:
                    expected_path = os.path.join(person_dir, fname)
                    if not os.path.exists(expected_path):
                        try:
                            gf = self.fs.get(ObjectId(img_id))
                            with open(expected_path, "wb") as f:
                                f.write(gf.read())
                            restored += 1
                            print(f"Restored file: {fname}")
                        except Exception as e:
                            print(f"Failed to restore file {fname}: {e}")
            
            else:
                fname, img_id = entries[0]
                expected_path = os.path.join(self.TRAIN_DIR, fname)
                
                if not os.path.exists(expected_path):
                    try:
                        gf = self.fs.get(ObjectId(img_id))
                        with open(expected_path, "wb") as f:
                            f.write(gf.read())
                        restored += 1
                        print(f"Restored single file: {fname}")
                    except Exception as e:
                        print(f"Failed to restore single file {fname}: {e}")

                if os.path.exists(person_dir):
                    for f in os.listdir(person_dir):
                        if f == fname:
                            try:
                                os.rename(os.path.join(person_dir, f), expected_path)
                                print(f"Moved single file '{fname}' out of folder.")
                            except Exception as e:
                                print(f"Failed to move single file out of folder: {e}")
                    if not os.listdir(person_dir):
                        try:
                            os.rmdir(person_dir)
                        except Exception as e:
                            print(f"Could not remove empty directory {person_dir}: {e}")

        orphans_removed = 0
        for root_dir, dirs, files in os.walk(self.TRAIN_DIR, topdown=False):
            for f in files:
                full_path = os.path.join(root_dir, f)
                if f not in filenames_in_db:
                    try:
                        os.remove(full_path)
                        orphans_removed += 1
                        print(f"Removed orphan file: {f}")
                    except Exception as e:
                        print(f"Could not remove local file {full_path}: {e}")
            
            for d in dirs:
                dir_path = os.path.join(root_dir, d)
                if not os.listdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                        print(f"Removed empty directory: {d}")
                    except Exception as e:
                        print(f"Could not remove empty directory {dir_path}: {e}")

        self.refresh_faces(self.app_state["face_list"])
        self.status(f"Sync complete. Restored {restored} from DB, removed {orphans_removed} local orphan(s).")

    def build_preview_panel(self, parent_frame, mode="idle"):
        for w in parent_frame.winfo_children():
            w.destroy()

        self.app_state["title_var"] = tk.StringVar()
        self.app_state["subtitle_var"] = tk.StringVar()
        
        ttk.Label(parent_frame, textvariable=self.app_state["title_var"], style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
        ttk.Label(parent_frame, textvariable=self.app_state["subtitle_var"], style="Subtitle.TLabel").pack(anchor="w", padx=10, pady=(0, 10))
        
        self.app_state["preview"] = ttk.Label(parent_frame, relief="solid", background="white")
        self.app_state["preview"].pack(fill="both", expand=True, padx=10, pady=10)

        bottom = ttk.Frame(parent_frame, style="Control.TFrame")
        bottom.pack(fill="x", pady=(0, 10), padx=10)

        if mode == "camera":
            ttk.Button(bottom, text="Capture", command=self.capture_from_camera, style="Action.TButton").pack(side="left", padx=5)
            ttk.Button(bottom, text="Cancel", command=self.cancel_camera, style="Action.TButton").pack(side="left", padx=5)
            self.app_state["title_var"].set("Live Camera")
            self.app_state["subtitle_var"].set("Click Capture")

        elif mode == "add_form":
            form_frame = ttk.Frame(bottom)
            form_frame.pack(fill="x", expand=True)
            ttk.Label(form_frame, text="Name:", style="Form.TLabel").pack(side="left", padx=(0, 6), pady=(5, 5))
            name_entry = ttk.Entry(form_frame)
            name_entry.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=(5, 5))
            ttk.Button(bottom, text="Save", command=lambda: self.save_face_inline(name_entry.get()), style="Action.TButton").pack(side="left", padx=5)
            ttk.Button(bottom, text="Cancel", command=lambda: (self.build_preview_panel(parent_frame, "idle"), self.clear_preview(), self.status("Add cancelled.")), style="Action.TButton").pack(side="left", padx=5)
            self.app_state["title_var"].set("New Face")
            self.app_state["subtitle_var"].set("Enter name and click Save")
        else:
            self.app_state["title_var"].set("No selection")
            self.app_state["subtitle_var"].set("")

    def on_list_double_click(self, face_list_widget, right_panel_widget):
        sel = face_list_widget.curselection()
        if not sel:
            return
        name = self.app_state["name_order"][sel[0]]
        docs = self.app_state["name_groups"].get(name, [])
        if not docs:
            self.clear_preview()
            return
        
        self.app_state["current_gallery_docs"] = docs
        self.app_state["selected_photo_indices"] = set()
        
        def on_photo_click(idx, e, parent_frame):
            if e.state & 0x4:
                if idx in self.app_state["selected_photo_indices"]:
                    self.app_state["selected_photo_indices"].remove(idx)
                else:
                    self.app_state["selected_photo_indices"].add(idx)
            else:
                self.app_state["selected_photo_indices"] = {idx}
            self.update_selection_visuals(parent_frame)
            self.update_delete_button_state()
            
        def update_selection_visuals(parent_frame):
            for i, widget in enumerate(parent_frame.winfo_children()):
                if i in self.app_state["selected_photo_indices"]:
                    widget.configure(style="Selected.Gallery.TFrame")
                else:
                    widget.configure(style="Unselected.Gallery.TFrame")
                    
        def update_delete_button_state():
            if self.app_state["delete_btn"]:
                if self.app_state["selected_photo_indices"]:
                    self.app_state["delete_btn"].state(["!disabled"])
                else:
                    self.app_state["delete_btn"].state(["disabled"])

        for w in right_panel_widget.winfo_children():
            w.destroy()

        ttk.Label(right_panel_widget, text=f"{name} ({len(docs)} photos)", style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 6))

        gallery_frame = ttk.Frame(right_panel_widget, style="Gallery.TFrame")
        gallery_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        canvas = tk.Canvas(gallery_frame, bg="white", highlightthickness=0)
        scroll_y = ttk.Scrollbar(gallery_frame, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas, style="Gallery.TFrame")
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(0, 0))
        scroll_y.pack(side="right", fill="y", padx=(0, 0))
        
        self.app_state["current_gallery_images"] = []

        for idx, doc in enumerate(docs):
            try:
                gf = self.fs.get(ObjectId(doc["image_id"]))
                img = self.pil_from_bytes(gf.read(), size=(200, 200))
                imgtk = ImageTk.PhotoImage(img)
                self.app_state["current_gallery_images"].append(imgtk)
                
                img_container = ttk.Frame(frame, style="Unselected.Gallery.TFrame")
                img_container.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)
                
                lbl = ttk.Label(img_container, image=imgtk)
                lbl.image = imgtk
                lbl.pack(padx=2, pady=2)
                
                img_container.bind("<Button-1>", lambda e, i=idx: on_photo_click(i, e, frame))
                
                def show_delete_menu(event, doc_to_delete):
                    menu = tk.Menu(self.root, tearoff=0, bg="#ffffff", fg="#333333")
                    menu.add_command(label="Delete Photo", command=lambda: self.delete_single_photo(doc_to_delete))
                    menu.tk_popup(event.x_root, event.y_root)
                
                img_container.bind("<Button-3>", lambda e, d=doc: show_delete_menu(e, d))
                
            except Exception as e:
                print(f"Could not load image {idx} for {name}: {e}")

        delete_controls_frame = ttk.Frame(gallery_frame, style="Control.TFrame")
        delete_controls_frame.pack(fill="x", side="bottom")
        self.app_state["delete_btn"] = ttk.Button(delete_controls_frame, text="Delete Selected Photos", command=self.delete_selected_photos, style="Action.TButton", state="disabled")
        self.app_state["delete_btn"].pack(padx=10, pady=10)

        self.status(f"Showing all {len(docs)} photos for {name}. Use Ctrl+Click to select multiple.")

    def delete_single_photo(self, doc):
        if not messagebox.askyesno("Delete Photo", "Are you sure you want to delete this photo?"):
            return
        
        try:
            if doc.get("image_id"):
                self.fs.delete(ObjectId(doc["image_id"]))
            self.coll.delete_one({"_id": doc["_id"]})
            
            fname = self.gridfs_filename_of(doc)
            if fname:
                name_part = os.path.splitext(fname.split("_")[0])[0]
                person_dir = os.path.join(self.TRAIN_DIR, name_part)
                file_path = os.path.join(person_dir, fname)
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                if os.path.exists(person_dir) and not os.listdir(person_dir):
                    os.rmdir(person_dir)

            self.status(f"Deleted photo for '{doc['name']}'. Syncing...")
            self.sync_db_and_folder()
            self.refresh_faces(self.app_state["face_list"])
            
        except Exception as e:
            self.status(f"Error deleting photo: {e}")

    def delete_selected_photos(self):
        selected_docs = [self.app_state["current_gallery_docs"][i] for i in self.app_state["selected_photo_indices"]]
        
        if not selected_docs:
            self.status("No photos selected to delete.")
            return
            
        if not messagebox.askyesno("Delete Photos", f"Are you sure you want to delete {len(selected_docs)} selected photos?"):
            return

        deleted_count = 0
        for doc in selected_docs:
            try:
                if doc.get("image_id"):
                    self.fs.delete(ObjectId(doc["image_id"]))
                self.coll.delete_one({"_id": doc["_id"]})
                
                fname = self.gridfs_filename_of(doc)
                if fname:
                    name_part = os.path.splitext(fname.split("_")[0])[0]
                    person_dir = os.path.join(self.TRAIN_DIR, name_part)
                    file_path = os.path.join(person_dir, fname)
                    if os.path.exists(file_path):
                        os.remove(file_path)

                deleted_count += 1
            except Exception as e:
                print(f"Error deleting photo for '{doc['name']}': {e}")
                
        self.status(f"Deleted {deleted_count} photos. Syncing...")
        self.sync_db_and_folder()
        self.refresh_faces(self.app_state["face_list"])

    def delete_selected(self):
        sel = self.app_state["face_list"].curselection()
        if not sel:
            self.status("Select a person to delete.")
            return
        name = self.app_state["name_order"][sel[0]]
        if not messagebox.askyesno("Delete", f"Delete ALL photos for '{name}' from DB and folder?"):
            return
        docs = list(self.coll.find({"name": name}))
        deleted_db_count = 0
        deleted_local_count = 0
        
        for d in docs:
            try:
                if d.get("image_id"):
                    self.fs.delete(ObjectId(d["image_id"]))
                    deleted_db_count += 1
            except Exception as e:
                print(f"Could not delete GridFS file for {name}: {e}")
                pass
            self.coll.delete_one({"_id": d["_id"]})

        person_dir = os.path.join(self.TRAIN_DIR, name)
        if os.path.exists(person_dir):
            for f in os.listdir(person_dir):
                try:
                    os.remove(os.path.join(person_dir, f))
                    deleted_local_count += 1
                except Exception as e:
                    print(f"Could not delete local file {f}: {e}")
                    pass
            try:
                os.rmdir(person_dir)
            except Exception as e:
                print(f"Could not remove directory {person_dir}: {e}")
                pass

        loose_file_path = os.path.join(self.TRAIN_DIR, f"{name}.jpg")
        if os.path.exists(loose_file_path):
            try:
                os.remove(loose_file_path)
                deleted_local_count += 1
            except Exception as e:
                print(f"Could not remove loose file {loose_file_path}: {e}")
                pass

        self.refresh_faces(self.app_state["face_list"])
        self.build_preview_panel(self.app_state["right_panel"], "idle")
        self.clear_preview()
        self.status(f"Deleted '{name}': {deleted_db_count} DB photos and {deleted_local_count} local files removed.")

    def begin_add_from_file(self):
        self.stop_camera_mode()
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if not file_path:
            self.status("Add cancelled.")
            return
        bgr = cv2.imread(file_path)
        if bgr is None:
            self.status("Could not read image.")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.show_add_form(rgb)

    def begin_add_from_camera(self):
        self.stop_camera_mode()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap or not cap.isOpened():
                self.status("Cannot access camera.")
                return
        self.app_state["cap"] = cap
        self.app_state["camera_running"] = True
        self.build_preview_panel(self.app_state["right_panel"], "camera")
        self.status("Camera started. Click Capture to take photo, or Cancel.")

        def loop():
            if not self.app_state["camera_running"]:
                return
            ok, frame = self.app_state["cap"].read()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = self.pil_from_np(frame_rgb, self.app_state["preview_size"])
                self.set_preview_from_pil(pil, title="Live Camera", subtitle="Click Capture")
                self.app_state["last_cam_frame_rgb"] = frame_rgb
            self.app_state["camera_loop_id"] = self.root.after(20, loop)
        loop()

    def capture_from_camera(self):
        rgb = self.app_state.get("last_cam_frame_rgb")
        if rgb is None:
            self.status("No frame available to capture.")
            return
        self.stop_camera_mode()
        self.show_add_form(rgb)

    def cancel_camera(self):
        self.stop_camera_mode()
        self.build_preview_panel(self.app_state["right_panel"], "idle")
        self.clear_preview()
        self.status("Camera cancelled.")

    def stop_camera_mode(self):
        if self.app_state.get("camera_loop_id"):
            try:
                self.root.after_cancel(self.app_state["camera_loop_id"])
            except Exception:
                pass
            self.app_state["camera_loop_id"] = None
        if self.app_state.get("cap") is not None:
            try:
                self.app_state["cap"].release()
            except Exception:
                pass
            self.app_state["cap"] = None
        self.app_state["camera_running"] = False
        self.app_state.pop("last_cam_frame_rgb", None)

    def show_add_form(self, rgb_image):
        self.app_state["pending_rgb_image"] = rgb_image
        self.build_preview_panel(self.app_state["right_panel"], "add_form")
        pil = self.pil_from_np(rgb_image, self.app_state["preview_size"])
        self.set_preview_from_pil(pil, title="New Face", subtitle="Enter name and click Save")

    def next_filename_for(self, name_upper: str) -> str:
        person_dir = os.path.join(self.TRAIN_DIR, name_upper)
        os.makedirs(person_dir, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(name_upper)}_(\d+)\.jpg$", re.IGNORECASE)
        max_n = 0
        for f in os.listdir(person_dir):
            m = pattern.match(f)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"{name_upper}_{max_n + 1}.jpg"

    def save_face_inline(self, name_input: str):
        rgb_image = self.app_state.get("pending_rgb_image")
        if rgb_image is None:
            self.status("No image to save.")
            return

        name = (name_input or "").strip()
        if not name:
            self.status("Name is required.")
            return
        name = name.upper()

        boxes = face_recognition.face_locations(rgb_image)
        encs = face_recognition.face_encodings(rgb_image, boxes)
        if not encs:
            self.status("No face detected. Try again.")
            return
        encoding = encs[0]

        pil = self.pil_from_np(rgb_image)
        bio = io.BytesIO()
        pil.save(bio, format="JPEG", quality=92)
        img_bytes = bio.getvalue()
        
        filename = f"{name}.jpg"
        person_dir = os.path.join(self.TRAIN_DIR, name)

        loose_file_exists = os.path.exists(os.path.join(self.TRAIN_DIR, filename))
        folder_exists = os.path.exists(person_dir)

        if not loose_file_exists and not folder_exists:
            image_id = self.fs.put(img_bytes, filename=filename)
            with open(os.path.join(self.TRAIN_DIR, filename), "wb") as f:
                f.write(img_bytes)
            
        else:
            os.makedirs(person_dir, exist_ok=True)
            
            if loose_file_exists:
                loose_file_doc = self.coll.find_one({"name": name, "encoding": {"$exists": True}})
                if loose_file_doc:
                    new_loose_file_name = f"{name}_1.jpg"
                    with open(os.path.join(self.TRAIN_DIR, filename), "rb") as f:
                        new_image_id = self.fs.put(f.read(), filename=new_loose_file_name)
                    
                    self.coll.update_one({"_id": loose_file_doc["_id"]}, {"$set": {"image_id": new_image_id}})
                    
                    os.rename(os.path.join(self.TRAIN_DIR, filename), os.path.join(person_dir, new_loose_file_name))

            filename = self.next_filename_for(name)
            image_id = self.fs.put(img_bytes, filename=filename)
            with open(os.path.join(person_dir, filename), "wb") as f:
                f.write(img_bytes)
        
        self.coll.insert_one({"name": name, "encoding": encoding.tolist(), "image_id": image_id})
        self.refresh_faces(self.app_state["face_list"])
        self.build_preview_panel(self.app_state["right_panel"], "idle")
        self.set_preview_from_pil(pil, title=name, subtitle=f"Saved as {filename}")
        self.status(f"Saved {name} → {filename}")

    def log_detections_from_feed(self, detections, full_frame):
        # Implement the logging logic here
        now = datetime.now()
        for det in detections:
            name = det['name']
            x1, y1, x2, y2 = det['coords']
            face_img = full_frame[y1:y2, x1:x2]
            
            # Check if this person was logged recently
            if name not in self.last_logged_time or (now - self.last_logged_time[name]).total_seconds() > self.app_state["LOG_INTERVAL"]:
                log_detection(name, face_img)
                self.last_logged_time[name] = now