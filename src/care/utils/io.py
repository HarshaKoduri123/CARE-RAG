from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")