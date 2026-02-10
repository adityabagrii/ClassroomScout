import json
import os
import re
from typing import Dict, List, Optional

URL_RE = re.compile(r"https?://[^\s\)\]]+", re.IGNORECASE)
DRIVE_FILE_RE = re.compile(r"https?://drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([A-Za-z0-9_-]{10,})")
DOCS_RE = re.compile(r"https?://docs\.google\.com/(document|presentation|spreadsheets)/d/([A-Za-z0-9_-]{10,})")


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out


def extract_json_block(s: str) -> Optional[Dict]:
    if not s:
        return None
    a = s.find("{")
    b = s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    cand = s[a : b + 1]
    try:
        return json.loads(cand)
    except Exception:
        return None


def nonempty(x: Optional[str]) -> bool:
    return bool(x and x.strip())


def extract_drive_ids_from_obj(obj) -> set:
    ids = set()

    def walk(o):
        if isinstance(o, dict):
            if "driveFile" in o and isinstance(o["driveFile"], dict):
                df = o["driveFile"].get("driveFile", {})
                fid = df.get("id")
                if fid:
                    ids.add(fid)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(obj)
    return ids


def extract_drive_ids_from_text(text: str) -> set:
    ids = set()
    if not text:
        return ids
    urls = URL_RE.findall(text)
    for u in urls:
        m = DRIVE_FILE_RE.search(u)
        if m:
            ids.add(m.group(1))
        m2 = DOCS_RE.search(u)
        if m2:
            ids.add(m2.group(2))
    return ids


def cache_path_for_drive(cache_dir: str, file_id: str, mime_type: str) -> str:
    if mime_type == "application/pdf":
        ext = ".pdf"
    elif mime_type == "application/vnd.google-apps.document":
        ext = ".gdoc.pdf"
    elif "presentation" in mime_type or mime_type == "application/vnd.ms-powerpoint":
        ext = ".pptx"
    elif "wordprocessingml" in mime_type or mime_type == "application/msword":
        ext = ".docx"
    else:
        ext = ".bin"
    return os.path.join(cache_dir, f"{file_id}{ext}")
