import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import sys
import os

class MainScriptLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Script Launcher")
        self.geometry("850x500")
        self.configure(bg="#0d1117")
        self.output_process = None

        self.style = ttk.Style(self)
        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Theme setup
        self.style.theme_use("clam")
        self.style.configure("Header.TFrame", background="#161b22")
        self.style.configure("TFrame", background="#0d1117")
        self.style.configure("TLabel", background="#161b22", foreground="white", font=("Segoe UI", 11, "bold"))
        self.style.configure("Green.TButton", background="#2ea043", foreground="white", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.map("Green.TButton", background=[("active", "#238636")])
        self.style.configure("Red.TButton", background="#dc3545", foreground="white", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.map("Red.TButton", background=[("active", "#b52a37")])
        self.style.configure("TCombobox", fieldbackground="#21262d", foreground="white")
        self.style.configure("TProgressbar", thickness=6, background="#2ea043")

        # Header bar
        header_frame = ttk.Frame(self, style="Header.TFrame", padding=10)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="🚀 Script Launcher", style="TLabel").pack(side=tk.LEFT)

        # Control frame
        control_frame = ttk.Frame(self, style="TFrame", padding=10)
        control_frame.pack(fill=tk.X)

        ttk.Label(control_frame, text="Select Script:", background="#0d1117", foreground="white").pack(side=tk.LEFT, padx=5)
        self.script_selector = ttk.Combobox(control_frame, values=["main.py", "face_viewer_gui.py"], state="readonly", width=30)
        self.script_selector.set("main.py")
        self.script_selector.pack(side=tk.LEFT, padx=5)

        self.run_button = ttk.Button(control_frame, text="▶ Run Script", style="Green.TButton", command=self.run_selected_script)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="■ Stop", style="Red.TButton", command=self.stop_script, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self, mode="determinate", style="TProgressbar")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # Console area
        self.output_text = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, state=tk.NORMAL,
            bg="#0d1117", fg="#00ff55", font=("Consolas", 11),
            relief=tk.FLAT, insertbackground="white"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.output_text.insert(tk.END, "Process finished.\n> Script output goes here...\n")
        self.output_text.configure(state=tk.DISABLED)

        # Status bar
        status_frame = ttk.Frame(self, style="Header.TFrame", padding=5)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.bottom_status_label = ttk.Label(status_frame, text="✅ Ready", background="#161b22", foreground="#2ea043", font=("Segoe UI", 10, "bold"))
        self.bottom_status_label.pack(side=tk.LEFT)

    def run_selected_script(self):
        script_name = self.script_selector.get()
        if not script_name:
            return

        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.DISABLED)

        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.script_selector.configure(state=tk.DISABLED)
        self.progress_bar.start(10)
        self.bottom_status_label.configure(text="⏳ Running...", foreground="white")

        self.thread = threading.Thread(target=self.run_subprocess, args=(script_name,), daemon=True)
        self.thread.start()

    def run_subprocess(self, script_name):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=script_dir
            )
            for line in iter(self.output_process.stdout.readline, ''):
                self.after(1, self.print_to_gui, line)
            for line in iter(self.output_process.stderr.readline, ''):
                self.after(1, self.print_to_gui, line)
            self.output_process.wait()
            self.after(1, self.on_process_complete)
        except Exception as e:
            self.after(1, self.print_to_gui, f"Error: {e}\n")
            self.after(1, self.on_process_complete)

    def print_to_gui(self, text):
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def stop_script(self):
        if self.output_process and self.output_process.poll() is None:
            self.output_process.terminate()
            self.bottom_status_label.configure(text="🛑 Script stopped", foreground="#dc3545")

    def on_process_complete(self):
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.script_selector.configure(state=tk.NORMAL)
        self.progress_bar.stop()
        self.bottom_status_label.configure(text="✅ Ready", foreground="#2ea043")

    def on_close(self):
        if self.output_process and self.output_process.poll() is None:
            self.output_process.terminate()
        self.destroy()


if __name__ == "__main__":
    app = MainScriptLauncher()
    app.mainloop()
