import sqlite3
import time
import json
from typing import Any, Dict, List, Optional

from .logging import log_msg

EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  course_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  item_id TEXT NOT NULL,
  item_ts INTEGER,
  title TEXT,
  text TEXT,
  raw_json TEXT NOT NULL,
  updated_ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_course_updated ON events(course_id, updated_ts);
"""

MATERIALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS course_materials (
  material_id TEXT PRIMARY KEY,
  course_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  drive_file_id TEXT,
  url TEXT,
  title TEXT,
  mime_type TEXT,
  md5_checksum TEXT,
  page_or_slide_count INTEGER,
  eligible INTEGER DEFAULT 0,
  status TEXT DEFAULT 'new',
  first_seen_ts INTEGER NOT NULL,
  last_seen_ts INTEGER NOT NULL,
  seen_in_json TEXT,
  summary_v1 TEXT,
  tags_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_materials_course ON course_materials(course_id);
"""

CHECKPOINTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_ts INTEGER NOT NULL
);
"""


def now_ts() -> int:
    return int(time.time())


class DB:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def init(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(EVENTS_SCHEMA)
            con.executescript(MATERIALS_SCHEMA)
            con.executescript(CHECKPOINTS_SCHEMA)
            con.commit()
        finally:
            con.close()
        log_msg(f"db.init db_path={self.db_path}")

    def _exec(self, sql: str, params=()):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()

    def _fetchone(self, sql: str, params=()):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            r = con.execute(sql, params).fetchone()
            return dict(r) if r else None
        finally:
            con.close()

    def _fetchall(self, sql: str, params=()):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(sql, params).fetchall()
            return [dict(x) for x in rows]
        finally:
            con.close()

    def set_checkpoint(self, key: str, value: str) -> None:
        self._exec(
            """
            INSERT INTO checkpoints(key,value,updated_ts)
            VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts
            """,
            (key, value, now_ts()),
        )

    def get_checkpoint(self, key: str, default: str = "0") -> str:
        row = self._fetchone("SELECT value FROM checkpoints WHERE key=?", (key,))
        return row["value"] if row else default

    def upsert_event(self, course_id: str, event_type: str, item_obj: Dict[str, Any]) -> str:
        item_id = item_obj.get("id") or ""
        if not item_id:
            raise ValueError("Event has no id")

        event_id = f"{event_type}:{item_id}"
        title = item_obj.get("title") or ""
        text = item_obj.get("text") or item_obj.get("description") or ""
        raw_json = json.dumps(item_obj, ensure_ascii=False)
        updated = now_ts()

        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """
                INSERT INTO events(event_id, course_id, event_type, item_id, item_ts, title, text, raw_json, updated_ts)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET
                  title=excluded.title,
                  text=excluded.text,
                  raw_json=excluded.raw_json,
                  updated_ts=excluded.updated_ts
                """,
                (event_id, course_id, event_type, item_id, None, title, text, raw_json, updated),
            )
            con.commit()
        finally:
            con.close()

        return event_id

    def get_recent_events(self, course_id: str, since_ts: int) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT * FROM events WHERE course_id=? AND updated_ts>=? ORDER BY updated_ts DESC
            """,
            (course_id, since_ts),
        )

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        return self._fetchone("SELECT * FROM events WHERE event_id=?", (event_id,))

    def get_all_events(self, course_id: str) -> List[Dict[str, Any]]:
        return self._fetchall(
            "SELECT event_id, raw_json, title, text FROM events WHERE course_id=?",
            (course_id,),
        )

    def upsert_material_from_meta(self, course_id: str, meta: Dict[str, Any], seen_event_ids: List[str]) -> str:
        material_id = f"drive:{meta['id']}"
        now = now_ts()
        seen_json = json.dumps(sorted(set(seen_event_ids)), ensure_ascii=False)

        row = self._fetchone("SELECT first_seen_ts FROM course_materials WHERE material_id=?", (material_id,))
        first_seen = row["first_seen_ts"] if row else now

        self._exec(
            """
            INSERT INTO course_materials(
              material_id, course_id, source_type, drive_file_id, url,
              title, mime_type, md5_checksum,
              page_or_slide_count, eligible, status,
              first_seen_ts, last_seen_ts, seen_in_json,
              summary_v1, tags_json
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(material_id) DO UPDATE SET
              title=excluded.title,
              mime_type=excluded.mime_type,
              md5_checksum=excluded.md5_checksum,
              last_seen_ts=excluded.last_seen_ts,
              seen_in_json=excluded.seen_in_json
            """,
            (
                material_id,
                course_id,
                "drive_file",
                meta["id"],
                None,
                meta.get("name"),
                meta.get("mimeType"),
                meta.get("md5Checksum"),
                None,
                0,
                "new",
                first_seen,
                now,
                seen_json,
                None,
                None,
            ),
        )
        return material_id

    def get_cached_summaries(self, course_id: str) -> List[Dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT material_id, title, summary_v1, tags_json
            FROM course_materials
            WHERE course_id=? AND status='cached' AND eligible=1
            """,
            (course_id,),
        )
        out = []
        for r in rows:
            try:
                r["tags"] = json.loads(r["tags_json"]) if r.get("tags_json") else []
            except Exception:
                r["tags"] = []
            out.append(r)
        return out

    def get_material(self, material_id: str) -> Optional[Dict[str, Any]]:
        return self._fetchone("SELECT * FROM course_materials WHERE material_id=?", (material_id,))

    def update_material_cache(self, material_id: str, count: int, eligible: bool, status: str) -> None:
        self._exec(
            """
            UPDATE course_materials
            SET page_or_slide_count=?, eligible=?, status=?, last_seen_ts=?
            WHERE material_id=?
            """,
            (count, 1 if eligible else 0, status, now_ts(), material_id),
        )

    def store_summary(self, material_id: str, summary_text: str, tags: List[str]) -> None:
        self._exec(
            """
            UPDATE course_materials
            SET summary_v1=?, tags_json=?, status='cached', last_seen_ts=?
            WHERE material_id=?
            """,
            (summary_text, json.dumps(tags, ensure_ascii=False), now_ts(), material_id),
        )

    def get_materials_missing_summary(self, course_id: str, limit: int) -> List[Dict[str, Any]]:
        return self._fetchall(
            """
            SELECT material_id FROM course_materials
            WHERE course_id=? AND summary_v1 IS NULL
            LIMIT ?
            """,
            (course_id, limit),
        )
