from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = Path("data/interim/pmc_patients_rag_full.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn