import os
import json
import time
from datetime import datetime
from pathlib import Path

class SessionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "sessions.jsonl"

    def _append(self, record: dict):
        """Appends a record to the JSONL file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"ERROR: Failed to write log: {e}")

    def log_session_start(self, session_id: str, ip: str = None):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": "SESSION_START",
            "session_id": session_id,
            "ip_address": ip or "unknown"
        }
        self._append(record)

    def log_event(self, session_id: str, event_type: str, details: dict = None):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "session_id": session_id,
            "details": details or {}
        }
        self._append(record)

    def log_session_end(self, session_id: str):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": "SESSION_END",
            "session_id": session_id
        }
        self._append(record)

# Singleton instance
session_logger = SessionLogger()
