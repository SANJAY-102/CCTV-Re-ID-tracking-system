"""
Persistent Database Manager for CCTV Tracking.
Saves employee identities, associated tracking sessions, and audit events to JSON or SQLite.
"""

import json
import os
from typing import Dict, Any, List, Optional
import time

from ..identity.employee import Employee
from ..config import DatabaseConfig


class Database:
    """
    Persistence layer for employee registries and tracking audit history.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        os.makedirs(self.config.storage_dir, exist_ok=True)
        self.file_path = os.path.join(self.config.storage_dir, self.config.db_name)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        if not os.path.exists(self.file_path):
            initial_data = {
                "version": "1.0",
                "created_at": time.time(),
                "employees": {},
                "audit_logs": []
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

    def save_employees(self, employees: Dict[str, Employee]):
        """Persist current employee registry to storage."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"version": "1.0", "created_at": time.time(), "employees": {}, "audit_logs": []}

        data["employees"] = {emp_id: emp.to_dict() for emp_id, emp in employees.items()}
        data["last_updated"] = time.time()

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Append an audit event log (e.g. REID_RECOVERY, NEW_ENROLLMENT)."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"version": "1.0", "created_at": time.time(), "employees": {}, "audit_logs": []}

        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        data.setdefault("audit_logs", []).append(event)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
