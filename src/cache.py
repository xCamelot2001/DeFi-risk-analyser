from __future__ import annotations
import os, json, time
from typing import Any, Optional

class Cache:
    def __init__(self, root: str = "data/cache", ttl_seconds: int = 1800):
        self.root = root
        self.ttl = ttl_seconds
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        if (time.time() - os.path.getmtime(path)) > self.ttl:
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        with open(path, "w") as f:
            json.dump(value, f)
