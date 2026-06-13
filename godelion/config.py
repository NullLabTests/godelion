import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_SEARCH_PATHS = [
    "./config.yaml",
    "./config.local.yaml",
    "~/.config/godelion/config.yaml",
    "~/.godelion.yaml",
]


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self._data: Dict[str, Any] = {}
        self._loaded_paths: list[str] = []

        if config_path:
            self._load_file(config_path)
        else:
            for path in CONFIG_SEARCH_PATHS:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    self._load_file(expanded)
                    if "local" in path or ".local" in path:
                        break  # local overrides stop further loading

        self._apply_env_overrides()

    def _load_file(self, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
            if data:
                self._deep_merge(self._data, data)
            self._loaded_paths.append(path)

    def _deep_merge(self, base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self):
        for key, value in os.environ.items():
            if key.startswith("GODELION_"):
                parts = key.lower().replace("godelion_", "").split("__")
                target = self._data
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        target = self._data
        for key in keys:
            if isinstance(target, dict):
                target = target.get(key)
            else:
                return default
            if target is None:
                return default
        return target if target is not None else default

    def set(self, value: Any, *keys: str):
        target = self._data
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value

    @property
    def loaded_paths(self) -> list[str]:
        return self._loaded_paths

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    @staticmethod
    def write_default(config_path: str = "config.yaml"):
        import pkgutil
        try:
            data = pkgutil.get_data(__package__ or "godelion", "config.yaml")
        except Exception:
            data = None
        if data:
            with open(config_path, "wb") as f:
                f.write(data)
        else:
            raise FileNotFoundError("Default config not found in package")


config = Config()
