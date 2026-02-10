import json
import time
from datetime import datetime
from typing import Any


def log_json(event: str, **fields: Any) -> None:
    payload = {"ts": int(time.time()), "event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False))


def log_msg(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
