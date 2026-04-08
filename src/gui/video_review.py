import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.settings import Settings
from src.service import Service
from src.cache import Cache


class VideoReviewApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dashcam Video Review")
        self.root.geometry("640x480")
        self.root.minsize(640, 480)

        self.settings = Settings()
        self.service = Service(self.settings.get("media_root", "dashcam"))
        self.cache = Cache(self.settings.get("cache_size", 100))

        self.current_category = "clips"
        self.current_items = []

        self._build_ui()
        self._refresh_categories()
        self._load_items()

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=8)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.root_label = ttk.Label(
            top_frame,
            text=f"Media Root: {self.service.media_root}"
        )
        self.root_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(top_frame, text="Choose Root", command=self.choose_root).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_frame, text="Refresh", command=self.refresh_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_frame, text="Settings", command=self.open_settings_window).pack(side=tk.LEFT, padx=4)

        main_frame = ttk.Frame(self.root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(main_frame, text="Folders", padding=8)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        self.category_listbox = tk.Listbox(left_frame, height=10, exportselection=False)
        self.category_listbox.pack(fill=tk.Y, expand=False)
        self.category_listbox.bind("<<ListboxSelect>>", self.on_category_select)

        right_frame = ttk.LabelFrame(main_frame, text="Items", padding=8)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ("name", "type")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings")
        self.tree.heading("name", text="Name")
        self.tree.heading("type", text="Type")
        self.tree.column("name", width=360, anchor="w")
        self.tree.column("type", width=100, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind("<Double-1>", self.open_selected_item)

        bottom_frame = ttk.Frame(self.root, padding=8)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(bottom_frame, text="Open Selected", command=self.open_selected_item).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom_frame, text="Open Folder", command=self.open_current_folder).pack(side=tk.LEFT, padx=4)

        self.status_label = ttk.Label(bottom_frame, text="Ready")
        self.status_label.pack(side=tk.RIGHT)

    def _refresh_categories(self):
        self.category_listbox.delete(0, tk.END)
        for category in self.service.get_categories():
            self.category_listbox.insert(tk.END, category)

        categories = self.service.get_categories()
        if self.current_category in categories:
            idx = categories.index(self.current_category)
            self.category_listbox.selection_clear(0, tk.END)
            self.category_listbox.selection_set(idx)
            self.category_listbox.activate(idx)
        elif categories:
            self.current_category = categories[0]
            self.category_listbox.selection_set(0)

    def _load_items(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        self.current_items = self.service.list_items(self.current_category)

        for item in self.current_items:
            item_type = self.service.get_item_type(item)
            self.tree.insert("", tk.END, values=(item.name, item_type))

        self.status_label.config(text=f"{len(self.current_items)} item(s)")

    def choose_root(self):
        selected = filedialog.askdirectory(title="Choose Media Root Folder")
        if not selected:
            return

        self.settings.set("media_root", selected)
        self.settings.save()

        self.service.set_media_root(selected)
        self.root_label.config(text=f"Media Root: {self.service.media_root}")
        self.refresh_all()

    def refresh_all(self):
        self.service.ensure_directories()
        self._refresh_categories()
        self._load_items()
        self.status_label.config(text="Refreshed")

    def on_category_select(self, event=None):
        selection = self.category_listbox.curselection()
        if not selection:
            return

        self.current_category = self.category_listbox.get(selection[0])
        self._load_items()

    def get_selected_path(self):
        selection = self.tree.selection()
        if not selection:
            return None

        index = self.tree.index(selection[0])
        if index < 0 or index >= len(self.current_items):
            return None

        return self.current_items[index]

    def open_selected_item(self, event=None):
        selected_path = self.get_selected_path()
        if selected_path is None:
            messagebox.showinfo("No Selection", "Please select a file or folder.")
            return

        try:
            self.service.open_path(selected_path)
            self.cache.add(str(selected_path))
            self.status_label.config(text=f"Opened: {selected_path.name}")
        except Exception as exc:
            messagebox.showerror("Open Error", f"Could not open item.\n\n{exc}")

    def open_current_folder(self):
        try:
            folder = self.service.get_category_path(self.current_category)
            self.service.open_path(folder)
            self.status_label.config(text=f"Opened folder: {folder.name}")
        except Exception as exc:
            messagebox.showerror("Open Error", f"Could not open folder.\n\n{exc}")

    def open_settings_window(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("380x420")
        settings_window.transient(self.root)
        settings_window.grab_set()

        frame = ttk.Frame(settings_window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Brightness").grid(row=0, column=0, sticky="w", pady=6)
        brightness_var = tk.IntVar(value=self.settings.get("brightness", 50))
        brightness_scale = ttk.Scale(frame, from_=0, to=100, orient="horizontal")
        brightness_scale.set(brightness_var.get())
        brightness_scale.grid(row=0, column=1, sticky="ew", pady=6)

        ttk.Label(frame, text="Clip Duration (sec)").grid(row=1, column=0, sticky="w", pady=6)
        clip_duration_var = tk.IntVar(value=self.settings.get("clip_duration", 30))
        ttk.Entry(frame, textvariable=clip_duration_var).grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(frame, text="Clip Quality").grid(row=2, column=0, sticky="w", pady=6)
        clip_quality_var = tk.StringVar(value=self.settings.get("clip_quality", "High"))
        ttk.Combobox(
            frame,
            textvariable=clip_quality_var,
            values=["Low", "Medium", "High"],
            state="readonly"
        ).grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(frame, text="Auto-Clip Duration (sec)").grid(row=3, column=0, sticky="w", pady=6)
        auto_clip_duration_var = tk.IntVar(value=self.settings.get("auto_clip_duration", 15))
        ttk.Entry(frame, textvariable=auto_clip_duration_var).grid(row=3, column=1, sticky="ew", pady=6)

        ttk.Label(frame, text="Auto-Clip Quality").grid(row=4, column=0, sticky="w", pady=6)
        auto_clip_quality_var = tk.StringVar(value=self.settings.get("auto_clip_quality", "Medium"))
        ttk.Combobox(
            frame,
            textvariable=auto_clip_quality_var,
            values=["Low", "Medium", "High"],
            state="readonly"
        ).grid(row=4, column=1, sticky="ew", pady=6)

        sudden_stops_var = tk.BooleanVar(value=self.settings.get("sudden_stops_enabled", True))
        ttk.Checkbutton(frame, text="Sudden Stops Enabled", variable=sudden_stops_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=6
        )

        warnings_var = tk.BooleanVar(value=self.settings.get("warnings_enabled", True))
        ttk.Checkbutton(frame, text="Warnings Enabled", variable=warnings_var).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=6
        )

        ttk.Label(frame, text="Cache Size").grid(row=7, column=0, sticky="w", pady=6)
        cache_size_var = tk.IntVar(value=self.settings.get("cache_size", 100))
        ttk.Entry(frame, textvariable=cache_size_var).grid(row=7, column=1, sticky="ew", pady=6)

        frame.columnconfigure(1, weight=1)

        def save_settings():
            try:
                self.settings.set("brightness", int(brightness_scale.get()))
                self.settings.set("clip_duration", int(clip_duration_var.get()))
                self.settings.set("clip_quality", clip_quality_var.get())
                self.settings.set("auto_clip_duration", int(auto_clip_duration_var.get()))
                self.settings.set("auto_clip_quality", auto_clip_quality_var.get())
                self.settings.set("sudden_stops_enabled", bool(sudden_stops_var.get()))
                self.settings.set("warnings_enabled", bool(warnings_var.get()))
                self.settings.set("cache_size", int(cache_size_var.get()))

                self.settings.save()
                self.cache.set_max_size(self.settings.get("cache_size", 100))

                messagebox.showinfo("Saved", "Settings saved successfully.")
                settings_window.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values where required.")

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=12)

        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=4)
