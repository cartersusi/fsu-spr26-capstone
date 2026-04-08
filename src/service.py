import os
import platform
import subprocess
from pathlib import Path


class Service:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    def __init__(self, media_root: str = "dashcam"):
        self.media_root = Path(media_root)
        self.required_dirs = {
            "clips": self.media_root / "clips",
            "photos": self.media_root / "photos",
            "long_form": self.media_root / "long_form",
        }
        self.ensure_directories()

    def set_media_root(self, media_root: str) -> None:
        self.media_root = Path(media_root)
        self.required_dirs = {
            "clips": self.media_root / "clips",
            "photos": self.media_root / "photos",
            "long_form": self.media_root / "long_form",
        }
        self.ensure_directories()

    def ensure_directories(self) -> None:
        for folder in self.required_dirs.values():
            folder.mkdir(parents=True, exist_ok=True)

    def get_categories(self):
        return list(self.required_dirs.keys())

    def get_category_path(self, category: str) -> Path:
        return self.required_dirs[category]

    def list_items(self, category: str):
        folder = self.get_category_path(category)
        if not folder.exists():
            return []
        items = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        return items

    def get_item_type(self, path: Path) -> str:
        if path.is_dir():
            return "Folder"
        suffix = path.suffix.lower()
        if suffix in self.VIDEO_EXTENSIONS:
            return "Video"
        if suffix in self.IMAGE_EXTENSIONS:
            return "Image"
        return "File"

    def open_path(self, path: Path) -> None:
        system_name = platform.system()
        if system_name == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system_name == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
