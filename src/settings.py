import json
from pathlib import Path


class Settings:
    DEFAULTS = {
        "media_root": "dashcam",
        "brightness": 50,
        "clip_duration": 30,
        "clip_quality": "High",
        "auto_clip_duration": 15,
        "auto_clip_quality": "Medium",
        "sudden_stops_enabled": True,
        "warnings_enabled": True,
        "cache_size": 100,
    }

    def __init__(self, config_path: str = "config/ui_settings.json"):
        self.config_path = Path(config_path)
        self.data = self.DEFAULTS.copy()
        self.load()

    def load(self) -> None:
        if self.config_path.exists():
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value

    def reset_defaults(self) -> None:
        self.data = self.DEFAULTS.copy()
        self.save()
