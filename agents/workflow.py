import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END

from .db import DB, now_ts
from .logging import log_msg
from .text_utils import (
    chunk_text,
    extract_json_block,
    nonempty,
    extract_drive_ids_from_obj,
    extract_drive_ids_from_text,
)
from .prompts import (
    ROUTER_PROMPT,
    QUIZ_DETECT_PROMPT,
    SYLLABUS_EXTRACT_PROMPT,
    ASSIGN_ANALYZE_PROMPT,
    SUMMARIZE_CHUNK_PROMPT,
    MERGE_SUMMARY_PROMPT,
    DIGEST_PROMPT,
    OUTLINE_PROMPT,
    STEP_EXPLAIN_PROMPT,
    QUIZ_TOPIC_EXPAND_PROMPT,
    QUIZ_SUBTOPIC_EXPLAIN_PROMPT,
    QUIZ_QA_CONCEPTUAL_PROMPT,
    QUIZ_QA_NUMERICAL_PROMPT,
    QUIZ_FORMAT_PROMPT,
    CODEGEN_STEP_PROMPT,
    CODEGEN_STITCH_PROMPT,
    FORMAT_DOC_PROMPT,
)


def keep_last(a, b):
    return b


class WFState(BaseModel):
    course_id: Annotated[str, keep_last]
    new_event_ids: Annotated[List[str], keep_last] = Field(default_factory=list)
    current_event_id: Annotated[Optional[str], keep_last] = None
    task_type: Annotated[Literal["quiz_flow", "assignment_flow", "digest_flow", "no_op", ""], keep_last] = ""
    topics_query: Annotated[Optional[str], keep_last] = None
    selected_material_ids: Annotated[List[str], keep_last] = Field(default_factory=list)
    doc_id: Annotated[Optional[str], keep_last] = None
    debug: Annotated[Dict[str, Any], keep_last] = Field(default_factory=dict)
    awaiting_user: Annotated[bool, keep_last] = False
    awaiting_kind: Annotated[Optional[Literal["syllabus"]], keep_last] = None
    awaiting_prompt: Annotated[Optional[str], keep_last] = None
    processed_event_ids: Annotated[List[str], keep_last] = Field(default_factory=list)
    user_feedback: Annotated[Optional[str], keep_last] = None


class Workflow:
    def __init__(self, settings, db: DB, tools: dict, agents: dict, allowed_mimes: set):
        self.settings = settings
        self.db = db
        self.tools = tools
        self.agents = agents
        self.allowed_mimes = allowed_mimes

    def _agent_last_text(self, result: dict) -> str:
        msgs = result.get("messages", [])
        if not msgs:
            return ""
        last = msgs[-1]
        if hasattr(last, "content") and last.content:
            return last.content
        ak = getattr(last, "additional_kwargs", {}) or {}
        rc = ak.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return rc
        return str(last)

    def _hitl_pending_key(self, course_id: str) -> str:
        return f"hitl:pending:{course_id}"

    def _hitl_payload_key(self, course_id: str, event_id: str) -> str:
        return f"hitl:payload:{course_id}:{event_id}"

    def _hitl_state_key(self, course_id: str, event_id: str) -> str:
        return f"hitl:state:{course_id}:{event_id}"

    def _get_pending_hitl(self, course_id: str) -> List[str]:
        raw = self.db.get_checkpoint(self._hitl_pending_key(course_id), "[]")
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _set_pending_hitl(self, course_id: str, pending: List[str]) -> None:
        self.db.set_checkpoint(self._hitl_pending_key(course_id), json.dumps(pending, ensure_ascii=False))

    def _store_hitl_payload(self, course_id: str, event_id: str, payload: Dict[str, Any]) -> None:
        self.db.set_checkpoint(self._hitl_payload_key(course_id, event_id), json.dumps(payload, ensure_ascii=False))

    def _load_hitl_payload(self, course_id: str, event_id: str) -> Optional[Dict[str, Any]]:
        raw = self.db.get_checkpoint(self._hitl_payload_key(course_id, event_id), "")
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _clear_hitl_payload(self, course_id: str, event_id: str) -> None:
        self.db.set_checkpoint(self._hitl_payload_key(course_id, event_id), "")

    def _add_pending_hitl(self, course_id: str, event_id: str, payload: Dict[str, Any]) -> None:
        pending = self._get_pending_hitl(course_id)
        if event_id not in pending:
            pending.append(event_id)
            self._set_pending_hitl(course_id, pending)
        self._store_hitl_payload(course_id, event_id, payload)

    def _save_hitl_state(self, course_id: str, event_id: str, state: WFState) -> None:
        self.db.set_checkpoint(self._hitl_state_key(course_id, event_id), json.dumps(state.model_dump(), ensure_ascii=False))

    def _load_hitl_state(self, course_id: str, event_id: str) -> Optional[WFState]:
        raw = self.db.get_checkpoint(self._hitl_state_key(course_id, event_id), "")
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return WFState(**obj)
        except Exception:
            return None
        return None

    def _clear_hitl_state(self, course_id: str, event_id: str) -> None:
        self.db.set_checkpoint(self._hitl_state_key(course_id, event_id), "")

    def _telegram_updates(self) -> List[Dict[str, Any]]:
        if "telegram_get_updates" not in self.tools:
            return []
        offset_key = "telegram:update_offset"
        offset_raw = self.db.get_checkpoint(offset_key, "0")
        try:
            offset = int(offset_raw or "0")
        except Exception:
            offset = 0
        resp = json.loads(self.tools["telegram_get_updates"].invoke({"offset": offset, "timeout": 5}))
        updates = resp.get("result", []) if isinstance(resp, dict) else []
        if updates:
            max_update_id = max(u.get("update_id", 0) for u in updates)
            self.db.set_checkpoint(offset_key, str(max_update_id + 1))
        return updates

    def _extract_hitl_reply(self, text: str, event_id: str) -> Optional[str]:
        token = f"[HITL:{event_id}]"
        if token not in text:
            return None
        return text.split(token, 1)[-1].strip()

    def _parse_feedback(self, text: str) -> Dict[str, Any]:
        t = text.strip()
        upper = t.upper()
        if upper.startswith("ACCEPT") or upper.startswith("APPROVE") or upper == "OK":
            return {"action": "accept", "feedback": ""}
        if upper.startswith("FEEDBACK:"):
            return {"action": "feedback", "feedback": t.split(":", 1)[-1].strip()}
        if upper.startswith("REGENERATE:"):
            return {"action": "feedback", "feedback": t.split(":", 1)[-1].strip()}
        return {"action": "feedback", "feedback": t}

    def _request_feedback(self, course_id: str, event_id: str, task_type: str, pdf_path: str, state: WFState) -> None:
        token = f"[HITL:{event_id}]"
        try:
            self.tools["telegram_send_document"].invoke({"file_path": pdf_path, "caption": f"{task_type} output"})
        except Exception as e:
            log_msg(f"telegram.send_document.failed error={e}")
        prompt = (
            f"{token}\nReview the PDF above.\n"
            "Reply with ACCEPT to finalize, or FEEDBACK: <your notes> to regenerate."
        )
        try:
            self.tools["notify_telegram"].invoke({"message": prompt})
        except Exception as e:
            log_msg(f"telegram.notify_failed error={e}")
        self._add_pending_hitl(course_id, event_id, {"kind": "feedback", "task_type": task_type})
        self._save_hitl_state(course_id, event_id, state)

    def _resolve_hitl(self, state: WFState) -> WFState:
        pending = self._get_pending_hitl(state.course_id)
        if not pending:
            return state
        updates = self._telegram_updates()
        if not updates:
            return state

        for ev_id in list(pending):
            payload = self._load_hitl_payload(state.course_id, ev_id) or {}
            kind = payload.get("kind")
            for u in updates:
                msg = u.get("message") or {}
                text = msg.get("text") or ""
                reply = self._extract_hitl_reply(text, ev_id)
                if not reply:
                    continue

                if kind == "syllabus":
                    state.topics_query = reply[:800]
                    state.current_event_id = ev_id
                    state.new_event_ids = [ev_id]
                    state.debug["force_task_type"] = "quiz_flow"
                    state.debug["from_hitl_syllabus"] = True
                    state.awaiting_user = False
                    state.awaiting_kind = None
                    state.awaiting_prompt = None

                elif kind == "feedback":
                    saved = self._load_hitl_state(state.course_id, ev_id)
                    if saved:
                        state = saved
                    fb = self._parse_feedback(reply)
                    if fb["action"] == "accept":
                        try:
                            self.tools["notify_telegram"].invoke({"message": f"[HITL:{ev_id}] Accepted."})
                        except Exception as e:
                            log_msg(f"telegram.notify_failed error={e}")
                        self._clear_hitl_state(state.course_id, ev_id)
                    else:
                        state.user_feedback = fb["feedback"]
                        state.debug["force_task_type"] = payload.get("task_type", "")
                        state.current_event_id = ev_id
                        state.new_event_ids = [ev_id]
                        state.debug["from_feedback"] = True
                        try:
                            self.tools["notify_telegram"].invoke(
                                {"message": f"[HITL:{ev_id}] Feedback received. Regenerating..."}
                            )
                        except Exception as e:
                            log_msg(f"telegram.notify_failed error={e}")

                pending.remove(ev_id)
                self._set_pending_hitl(state.course_id, pending)
                self._clear_hitl_payload(state.course_id, ev_id)
                return state

        return state

    def parse_due_date(self, ev: dict):
        due = ev.get("dueDate")
        time = ev.get("dueTime") or {}
        if not due:
            return None
        try:
            dt = datetime(
                due.get("year"),
                due.get("month"),
                due.get("day"),
                time.get("hours", 0),
                time.get("minutes", 0),
                time.get("seconds", 0),
                tzinfo=timezone.utc,
            )
            return dt
        except Exception:
            return None

    def minimal_event_context(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_type": ev.get("event_type", ""),
            "title": (ev.get("title") or "")[:200],
            "text": (ev.get("text") or "")[:2000],
        }

    def poller_node(self, state: WFState) -> WFState:
        state = self._resolve_hitl(state)
        ck_key = f"poll:last_ts:{state.course_id}"
        last_ts = int(self.db.get_checkpoint(ck_key, "0") or "0")

        anns = json.loads(self.tools["classroom_list_announcements"].invoke({"course_id": state.course_id}))
        cw = json.loads(self.tools["classroom_list_coursework"].invoke({"course_id": state.course_id}))
        mats = json.loads(self.tools["classroom_list_materials"].invoke({"course_id": state.course_id}))

        for a in anns.get("announcements", []):
            self.db.upsert_event(state.course_id, "announcement", a)
        for c in cw.get("courseWork", []):
            self.db.upsert_event(state.course_id, "coursework", c)
        for m in mats.get("courseWorkMaterial", []):
            self.db.upsert_event(state.course_id, "material", m)

        recent = self.db.get_recent_events(state.course_id, last_ts + 1)
        recent_ids = [r["event_id"] for r in recent]
        if state.debug.get("force_task_type") and state.current_event_id:
            if state.current_event_id not in recent_ids:
                recent_ids.insert(0, state.current_event_id)
        state.new_event_ids = recent_ids
        self.db.set_checkpoint(ck_key, str(now_ts()))

        log_msg(f"poller.done course_id={state.course_id} new_events={len(state.new_event_ids)}")
        state.debug["poller"] = {"new_events": state.new_event_ids[:10]}
        return state

    def router_node(self, state: WFState) -> WFState:
        if state.debug.get("force_task_type") and state.current_event_id:
            state.task_type = state.debug.get("force_task_type")
            return state
        if not state.new_event_ids:
            state.task_type = "no_op"
            return state

        for ev_id in state.new_event_ids:
            if ev_id in state.processed_event_ids:
                continue
            state.current_event_id = ev_id
            ev = self.db.get_event(state.current_event_id)
            if not ev:
                continue

            ctx = self.minimal_event_context(ev)
            prompt = ROUTER_PROMPT.format(**ctx)
            result = self.agents["router_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
            raw = self._agent_last_text(result)
            obj = extract_json_block(raw) or {}

            tt = obj.get("task_type", "no_op")
            if tt not in ("quiz_flow", "assignment_flow", "digest_flow", "no_op"):
                tt = "no_op"

            log_msg(
                f"router.decide task_type={tt} confidence={obj.get('confidence',0)} "
                f"title={ctx.get('title','')[:80]}"
            )

            if tt != "no_op":
                state.task_type = tt
                state.debug["router"] = obj
                return state

        state.task_type = "no_op"
        return state

    def quiz_detector_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            state.task_type = "no_op"
            return state
        ctx = self.minimal_event_context(ev)

        prompt = QUIZ_DETECT_PROMPT.format(title=ctx["title"], text=ctx["text"])
        result = self.agents["quiz_detector_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
        raw = self._agent_last_text(result)
        obj = extract_json_block(raw) or {}

        if obj.get("type") != "quiz" and obj.get("confidence", 0.0) >= 0.6:
            state.task_type = "no_op"

        if obj.get("syllabus_present") and nonempty(obj.get("syllabus_text")):
            state.topics_query = obj["syllabus_text"][:500]

        state.debug["quiz_detector"] = obj
        log_msg(f"quiz_detector.done type={obj.get('type')} confidence={obj.get('confidence',0)}")
        return state

    def syllabus_extractor_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            state.task_type = "no_op"
            return state

        if state.debug.get("from_hitl_syllabus") and state.topics_query:
            state.awaiting_user = False
            state.awaiting_kind = None
            state.awaiting_prompt = None
            return state

        ctx = self.minimal_event_context(ev)

        since = now_ts() - int(48 * 3600)
        near = self.db.get_recent_events(state.course_id, since)[:8]
        nearby_text = "\n".join(
            [f"- {(x.get('title') or '')[:120]}: {(x.get('text') or '')[:280]}" for x in near]
        )

        prompt = SYLLABUS_EXTRACT_PROMPT.format(title=ctx["title"], text=ctx["text"], nearby=nearby_text)
        result = self.agents["syllabus_extractor_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
        raw = self._agent_last_text(result)
        obj = extract_json_block(raw) or {}
        state.debug["syllabus_extractor"] = obj

        if obj.get("syllabus") == "missing":
            token = f"[HITL:{state.current_event_id}]"
            prompt = (
                f"{token}\nSyllabus missing for quiz event:\n"
                f"- Title: {ctx['title']}\n\n"
                "Reply with syllabus/topics (comma separated or multiline)."
            )
            try:
                self.tools["notify_telegram"].invoke({"message": prompt})
            except Exception as e:
                log_msg(f"syllabus.missing.notify_failed error={e}")

            self._add_pending_hitl(
                state.course_id,
                state.current_event_id,
                {"kind": "syllabus", "task_type": "quiz_flow"},
            )
            state.awaiting_user = True
            state.awaiting_kind = "syllabus"
            state.awaiting_prompt = prompt
            state.debug["awaiting_hitl"] = True
            log_msg(f"syllabus.missing.hitl event_id={state.current_event_id}")
            return state

        topics = obj.get("topics", [])
        if isinstance(topics, list) and topics:
            state.topics_query = ", ".join([t for t in topics if t])[:600]
        elif nonempty(obj.get("syllabus")):
            state.topics_query = str(obj["syllabus"])[:600]
        else:
            state.topics_query = ctx["title"][:200]

        state.awaiting_user = False
        state.awaiting_kind = None
        state.awaiting_prompt = None
        log_msg(f"syllabus.extracted topics_query={state.topics_query}")
        return state

    def human_input_syllabus_node(self, state: WFState) -> WFState:
        if not (state.awaiting_user and state.awaiting_kind == "syllabus"):
            return state

        print("\n" + (state.awaiting_prompt or "Please enter syllabus/topics:"))
        user_text = input("> ").strip()

        while not user_text:
            print("Empty input. Please paste syllabus/topics (or type a short topic list).")
            user_text = input("> ").strip()

        state.topics_query = user_text[:800]
        state.awaiting_user = False
        state.awaiting_kind = None
        state.awaiting_prompt = None

        log_msg(f"hitl.syllabus.provided topics_query={state.topics_query[:200]}")
        return state

    def assignment_analyzer_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            state.task_type = "no_op"
            return state
        ctx = self.minimal_event_context(ev)
        due_dt = self.parse_due_date(ev)
        if due_dt and datetime.now(timezone.utc) > due_dt:
            log_msg(f"assignment.skipped.overdue due={due_dt.isoformat()}")
            state.task_type = "no_op"
            return state

        prompt = ASSIGN_ANALYZE_PROMPT.format(title=ctx["title"], text=ctx["text"])
        result = self.agents["assignment_analyzer_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
        raw = self._agent_last_text(result)
        obj = extract_json_block(raw) or {}

        reqs = obj.get("requirements", [])
        if isinstance(reqs, list) and reqs:
            state.topics_query = ", ".join(reqs[:12])[:700]
        else:
            state.topics_query = ctx["title"][:200]

        state.debug["assignment_analyzer"] = obj
        log_msg(f"assignment.analyzed coding={obj.get('coding_or_non_coding','unknown')}")
        return state

    def retrieval_node(self, state: WFState) -> WFState:
        q = state.topics_query or "course topics"
        docs_ = self.db.get_cached_summaries(state.course_id)

        scored = []
        for d in docs_:
            sc = self.score_summary(d.get("summary_v1") or "", q)
            bonus = 0.0
            for t in d.get("tags", []):
                if t and t.lower() in q.lower():
                    bonus += 0.05
            scored.append((sc + bonus, d))
        scored.sort(key=lambda x: x[0], reverse=True)

        hits = []
        for sc, d in scored[: self.settings.top_k_retrieval]:
            hits.append(
                {
                    "material_id": d["material_id"],
                    "title": d.get("title") or "",
                    "score": round(sc, 4),
                    "tags": d.get("tags", [])[:8],
                    "preview": (d.get("summary_v1") or "")[:320],
                }
            )

        state.selected_material_ids = [h["material_id"] for h in hits]
        state.debug["retrieval"] = {"query": q, "hits": hits}
        log_msg(f"retrieval.done query={q} hits={len(hits)}")
        return state

    def score_summary(self, summary: str, query: str) -> float:
        if not summary or not query:
            return 0.0
        q = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        s = re.findall(r"[a-zA-Z0-9]+", summary.lower())
        if not s:
            return 0.0
        hit = sum(1 for w in s if w in q)
        return hit / max(1, len(s))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def summarize_one_chunk(self, title: str, chunk: str) -> Dict[str, Any]:
        preview = chunk[:240].replace("\n", " ")
        prompt = SUMMARIZE_CHUNK_PROMPT.format(title=title, preview=preview, chunk=chunk)
        raw = self._agent_last_text(
            self.agents["material_summarizer_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
        )
        obj = extract_json_block(raw)
        if not obj or not nonempty(obj.get("summary")):
            raise ValueError("chunk summary parse failed/empty")
        return obj

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def merge_summaries(self, title: str, chunk_summaries: List[str]) -> Dict[str, Any]:
        joined = "\n\n".join(chunk_summaries)[:16000]
        prompt = MERGE_SUMMARY_PROMPT.format(title=title, chunk_summaries=joined)
        raw = self._agent_last_text(
            self.agents["material_summarizer_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
        )
        obj = extract_json_block(raw)
        if not obj or not nonempty(obj.get("final_summary")):
            raise ValueError("merge parse failed/empty")
        if "tags" not in obj or not isinstance(obj["tags"], list):
            obj["tags"] = []
        return obj

    def ensure_summaries_node(self, state: WFState) -> WFState:
        ensured = []
        for mid in state.selected_material_ids:
            m = self.db.get_material(mid)
            if not m:
                continue
            if nonempty(m.get("summary_v1")):
                ensured.append(mid)
                continue

            file_id = m.get("drive_file_id")
            mime_type = m.get("mime_type") or ""
            title = (m.get("title") or mid)[:120]

            if mime_type not in self.allowed_mimes:
                self.db.update_material_cache(mid, 0, False, "skipped_type")
                continue

            path = self.tools["safe_cache_path_for_drive"](file_id, mime_type)
            if mime_type == "application/vnd.google-apps.document":
                self.tools["drive_export_google_doc"].invoke(
                    {"file_id": file_id, "out_path": path, "mime": "application/pdf"}
                )
                effective_mime = "application/pdf"
            else:
                self.tools["drive_download"].invoke({"file_id": file_id, "out_path": path})
                effective_mime = mime_type

            fx = json.loads(self.tools["file_count_and_extract"].invoke({"path": path, "mime_type": effective_mime}))
            count = int(fx.get("count", 0))
            text = fx.get("text", "") or ""

            eligible = count < self.settings.max_pages_or_slides
            status = "cached" if eligible else "skipped_size"
            self.db.update_material_cache(mid, count, eligible, status)

            if not eligible:
                log_msg(f"material.too_large material_id={mid} count={count}")
                continue

            if not nonempty(text):
                self.db.store_summary(mid, f"(No extractable text) {title}", [])
                ensured.append(mid)
                continue

            chunks = chunk_text(text, self.settings.max_chars, self.settings.overlap)[: self.settings.max_chunks]
            log_msg(f"summarize.start material_id={mid} chunks={len(chunks)}")

            chunk_summaries = []
            key_terms = []
            for i, ch in enumerate(chunks):
                try:
                    out = self.summarize_one_chunk(title, ch)
                    chunk_summaries.append(out["summary"])
                    key_terms.extend(out.get("key_terms", []))
                    log_msg(f"summarize.chunk_ok material_id={mid} chunk={i} chars={len(ch)}")
                except Exception as e:
                    snippet = ch[:500].replace("\n", " ")
                    chunk_summaries.append(f"(Fallback snippet) {snippet}")
                    log_msg(f"summarize.chunk_fail material_id={mid} chunk={i} error={str(e)}")

            try:
                merged = self.merge_summaries(title, chunk_summaries)
                final_summary = merged["final_summary"].strip()
                tags = [t.strip() for t in merged.get("tags", []) if t and t.strip()]
            except Exception as e:
                final_summary = "\n\n".join(chunk_summaries).strip()
                tags = sorted(set([t.strip() for t in key_terms if t and t.strip()]))[:20]
                log_msg(f"summarize.merge_fail material_id={mid} error={str(e)}")

            if not nonempty(final_summary):
                final_summary = f"(Empty summary fallback) {title}"

            self.db.store_summary(mid, final_summary, tags[:20])
            ensured.append(mid)
            log_msg(f"summarize.done material_id={mid} tags={len(tags)}")

        state.debug["ensure_summaries"] = {"ensured": ensured}
        log_msg(f"summaries.ensured count={len(ensured)}")
        return state

    def material_cache_node(self, state: WFState) -> WFState:
        events = self.db.get_all_events(state.course_id)
        drive_ids = set()
        seen_map = {}
        for ev in events:
            ev_id = ev["event_id"]
            raw = json.loads(ev.get("raw_json") or "{}")
            ids = extract_drive_ids_from_obj(raw)
            ids |= extract_drive_ids_from_text(ev.get("text") or "")
            ids |= extract_drive_ids_from_text(ev.get("title") or "")
            for fid in ids:
                drive_ids.add(fid)
                seen_map.setdefault(fid, []).append(ev_id)

        upserted = 0
        for fid in drive_ids:
            try:
                meta = json.loads(self.tools["drive_get_metadata"].invoke({"file_id": fid}))["meta"]
                if meta.get("mimeType") not in self.allowed_mimes:
                    continue
                self.db.upsert_material_from_meta(state.course_id, meta, seen_map.get(fid, []))
                upserted += 1
            except Exception:
                continue

        rows = self.db.get_materials_missing_summary(state.course_id, self.settings.max_summarize_per_run)
        state.selected_material_ids = [r["material_id"] for r in rows]
        state = self.ensure_summaries_node(state)

        state.debug["material_cache"] = {"drive_ids": len(drive_ids), "upserted": upserted}
        return state

    def outline_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            return state
        ctx = self.minimal_event_context(ev)
        analyzer = state.debug.get("assignment_analyzer", {})
        reqs = analyzer.get("requirements", [])
        dels = analyzer.get("deliverables", [])

        prompt = OUTLINE_PROMPT.format(
            title=ctx.get("title", ""),
            text=ctx.get("text", ""),
            requirements=", ".join(reqs)[:800],
            deliverables=", ".join(dels)[:800],
            feedback=state.user_feedback or "",
        )
        result = self.agents["outline_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
        raw = self._agent_last_text(result)
        obj = extract_json_block(raw) or {}

        if not obj or not obj.get("outline"):
            outline = []
            if reqs:
                outline.append("Requirements")
            if dels:
                outline.append("Deliverables")
            outline.append("Approach")
            outline.append("Evaluation")
            steps = []
            if reqs:
                steps.append("List requirements and clarify unknowns")
            steps.append("Plan approach and data preprocessing")
            steps.append("Implement baseline model")
            steps.append("Evaluate and iterate")
            if dels:
                steps.append("Prepare deliverables and verify submission")
            obj = {"outline": outline, "steps": steps, "assumptions": []}
        state.debug["outline"] = obj
        log_msg(f"outline.generated sections={len(obj.get('outline', []) if isinstance(obj, dict) else [])}")
        return state

    def step_explain_node(self, state: WFState) -> WFState:
        outline = state.debug.get("outline", {}) or {}
        steps = outline.get("steps", []) or []
        if not steps:
            state.debug["step_explanations"] = []
            return state
        ctx = state.debug.get("assignment_analyzer", {}) or {}
        context = json.dumps(ctx, ensure_ascii=False)

        explanations = []
        count = 1
        for s in steps:
            prompt = STEP_EXPLAIN_PROMPT.format(step=s, context=context)
            result = self.agents["step_explainer_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
            raw = self._agent_last_text(result)
            obj = extract_json_block(raw) or {}
            exp = obj.get("explanation") or raw
            explanations.append(exp.strip())
            log_msg(f"step_explain.done count={count}")
            count += 1

        state.debug["step_explanations"] = explanations
        return state

    def quiz_topic_expand_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            state.task_type = "no_op"
            return state

        topics = []
        if state.topics_query:
            topics = [t.strip() for t in state.topics_query.split(",") if t.strip()]
        if not topics:
            topics = [(ev.get("title") or "Quiz")]

        context = json.dumps(
            {"event": self.minimal_event_context(ev), "retrieval": state.debug.get("retrieval", {})},
            ensure_ascii=False,
        )

        expanded = []
        for t in topics:
            prompt = QUIZ_TOPIC_EXPAND_PROMPT.format(topic=t, context=context, feedback=state.user_feedback or "")
            res = self.agents["quiz_topic_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
            raw = self._agent_last_text(res)
            obj = extract_json_block(raw) or {}
            if obj:
                expanded.append(obj)

        state.debug["quiz_topic_expansion"] = {"topics": topics, "expanded": expanded}
        log_msg(f"quiz_topic_expand.done topics={len(topics)}")
        return state

    def quiz_subtopic_explain_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            state.task_type = "no_op"
            return state

        exp = state.debug.get("quiz_topic_expansion", {})
        expanded = exp.get("expanded", []) or []
        subtopics = []
        for item in expanded:
            for s in item.get("subtopics", []) or []:
                if s and s not in subtopics:
                    subtopics.append(s)

        if not subtopics:
            subtopics = exp.get("topics", []) or []

        context = json.dumps(
            {"event": self.minimal_event_context(ev), "retrieval": state.debug.get("retrieval", {})},
            ensure_ascii=False,
        )

        explanations = []
        for s in subtopics:
            prompt = QUIZ_SUBTOPIC_EXPLAIN_PROMPT.format(subtopic=s, context=context, feedback=state.user_feedback or "")
            res = self.agents["quiz_topic_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
            raw = self._agent_last_text(res)
            obj = extract_json_block(raw) or {}
            if obj:
                explanations.append(obj)

        state.debug["quiz_subtopic_explanations"] = {"subtopics": subtopics, "explanations": explanations}
        log_msg(f"quiz_subtopic_explain.done subtopics={len(subtopics)} explanations={len(explanations)}")
        return state

    def _qa_topup(self, questions: List[Dict[str, Any]], min_count: int, max_count: int, qa_prompt: str) -> List[Dict[str, Any]]:
        if len(questions) >= min_count:
            return questions[:max_count]
        res = self.agents["quiz_qa_agent"].invoke({"messages": [{"role": "user", "content": qa_prompt}]})
        raw = self._agent_last_text(res)
        obj = extract_json_block(raw) or {}
        more = obj.get("questions", []) if isinstance(obj, dict) else []
        for q in more:
            if q not in questions:
                questions.append(q)
        return questions[:max_count]

    def quiz_qa_node(self, state: WFState) -> WFState:
        exp = state.debug.get("quiz_topic_expansion", {})
        subexp = state.debug.get("quiz_subtopic_explanations", {})
        topics = exp.get("topics", []) or []
        subtopics = subexp.get("subtopics", []) or []

        topic_blob = json.dumps({"topics": topics, "subtopics": subtopics}, ensure_ascii=False)
        context = json.dumps(
            {"retrieval": state.debug.get("retrieval", {})},
            ensure_ascii=False,
        )

        conceptual_prompt = QUIZ_QA_CONCEPTUAL_PROMPT.format(topics=topic_blob, context=context)
        res1 = self.agents["quiz_qa_agent"].invoke({"messages": [{"role": "user", "content": conceptual_prompt}]})
        raw1 = self._agent_last_text(res1)
        obj1 = extract_json_block(raw1) or {}
        conceptual = obj1.get("questions", []) if isinstance(obj1, dict) else []
        conceptual = self._qa_topup(conceptual, 15, 20, conceptual_prompt)

        numerical_prompt = QUIZ_QA_NUMERICAL_PROMPT.format(topics=topic_blob, context=context)
        res2 = self.agents["quiz_qa_agent"].invoke({"messages": [{"role": "user", "content": numerical_prompt}]})
        raw2 = self._agent_last_text(res2)
        obj2 = extract_json_block(raw2) or {}
        numerical = obj2.get("questions", []) if isinstance(obj2, dict) else []
        numerical = self._qa_topup(numerical, 5, 10, numerical_prompt)

        state.debug["quiz_qa"] = {"conceptual": conceptual, "numerical": numerical}
        log_msg(f"quiz_qa.done conceptual={len(conceptual)} numerical={len(numerical)}")
        return state

    def format_markdown_chunked(self, raw_text: str) -> str:
        if not raw_text.strip():
            return raw_text
        chunk_size = 6000
        overlap = 300
        chunks = []
        i = 0
        n = len(raw_text)
        while i < n:
            j = min(n, i + chunk_size)
            chunks.append(raw_text[i:j])
            if j == n:
                break
            i = max(0, j - overlap)

        formatted_parts = []
        for ch in chunks:
            prompt = FORMAT_DOC_PROMPT.format(raw=ch)
            result = self.agents["formatter_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
            md = self._agent_last_text(result)
            if not md.strip():
                md = ch
            formatted_parts.append(md)

        return "".join(formatted_parts)

    def fix_unbalanced_backticks(self, md_text: str) -> str:
        count = md_text.count("```")
        if count % 2 != 0:
            idx = md_text.rfind("```")
            if idx != -1:
                md_text = md_text[:idx] + md_text[idx + 3 :]
        return md_text

    def quiz_doc_writer_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            state.task_type = "no_op"
            return state
        title = (ev.get("title") or state.current_event_id)[:80]
        doc_title = f"[Auto] Quiz Prep - {title}"

        exp = state.debug.get("quiz_topic_expansion", {})
        subexp = state.debug.get("quiz_subtopic_explanations", {})
        qa = state.debug.get("quiz_qa", {})

        md = []
        md.append(f"# {doc_title}\n")
        md.append("## Event\n")
        md.append(f"**Title:** {ev.get('title')}\n\n")
        md.append("**Text:**\n\n")
        md.append(f"{(ev.get('text') or '')[:4000]}\n\n")

        md.append("## Topics\n")
        for t in exp.get("topics", []) or []:
            md.append(f"- {t}\n")
        md.append("\n")

        md.append("## Topic Expansion\n")
        for item in exp.get("expanded", []) or []:
            md.append(f"### {item.get('topic','Topic')}\n")
            if item.get("learning_objectives"):
                md.append("**Things to Learn:**\n")
                for lo in item.get("learning_objectives", [])[:12]:
                    md.append(f"- {lo}\n")
            if item.get("subtopics"):
                md.append("**Subtopics:**\n")
                for st in item.get("subtopics", [])[:15]:
                    md.append(f"- {st}\n")
            md.append("\n")

        md.append("## Subtopic Explanations\n")
        for obj in subexp.get("explanations", []) or []:
            md.append(f"### {obj.get('subtopic','Subtopic')}\n")
            md.append((obj.get("explanation") or "") + "\n\n")
            if obj.get("key_points"):
                md.append("**Key Points:**\n")
                for kp in obj.get("key_points", [])[:10]:
                    md.append(f"- {kp}\n")
                md.append("\n")
            if obj.get("formulae"):
                md.append("**Formulae:**\n")
                for f in obj.get("formulae", [])[:8]:
                    md.append(f"- {f}\n")
                md.append("\n")
            if obj.get("example"):
                md.append("**Example:**\n")
                md.append(obj.get("example") + "\n\n")

        md.append("## Practice Questions (Conceptual)\n")
        for q in qa.get("conceptual", []) or []:
            md.append(f"**Q:** {q.get('question','')}\n")
            md.append(f"**A:** {q.get('answer','')}\n\n")

        md.append("## Practice Questions (Numerical)\n")
        for q in qa.get("numerical", []) or []:
            md.append(f"**Q:** {q.get('question','')}\n")
            md.append(f"**A:** {q.get('answer','')}\n\n")

        raw_text = "".join(md)
        md_text = self.format_markdown_chunked(raw_text)
        md_text = self.fix_unbalanced_backticks(md_text)

        os.makedirs(self.settings.cache_dir, exist_ok=True)
        safe_id = state.current_event_id.replace(":", "_")
        md_path = os.path.join(self.settings.cache_dir, f"{safe_id}_quiz.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        pdf_path = os.path.join(self.settings.cache_dir, f"{safe_id}_quiz.pdf")
        ok = self.tools["render_markdown_to_pdf"](md_path, pdf_path)

        if ok and os.path.exists(pdf_path):
            state.debug["doc_writer"] = {"md_path": md_path, "pdf_path": pdf_path}
            log_msg(f"doc.created local_md={md_path} local_pdf={pdf_path}")
            self._request_feedback(state.course_id, state.current_event_id, "quiz", pdf_path, state)
        else:
            log_msg("doc.create.failed")
        return state

    def strip_outer_code_fence(self, md_text: str) -> str:
        s = md_text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines[0].startswith("```") and lines[-1].startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return md_text

    def doc_writer_node(self, state: WFState) -> WFState:
        ev = self.db.get_event(state.current_event_id)
        if not ev:
            state.task_type = "no_op"
            return state
        log_msg(f"doc_writer.start event_id={state.current_event_id}")
        title = (ev.get("title") or state.current_event_id)[:80]

        doc_title = f"[Auto] {state.task_type.replace('_flow','').title()} - {title}"

        retrieval_info = state.debug.get("retrieval", {})
        outline_obj = state.debug.get("outline", {}) or {}
        steps = outline_obj.get("steps", []) or []
        step_explanations = state.debug.get("step_explanations", []) or []

        md = []
        md.append(f"# {doc_title}\n")
        md.append("## Event\n")
        md.append(f"**Type:** {ev.get('event_type')}\n\n")
        md.append(f"**Title:** {ev.get('title')}\n\n")
        md.append("**Text:**\n\n")
        md.append(f"{(ev.get('text') or '')[:4000]}\n\n")

        md.append("## Assignment Analysis\n")
        md.append("```json\n" + json.dumps(state.debug.get("assignment_analyzer", {}), ensure_ascii=False, indent=2) + "\n```\n\n")

        md.append("## Outline\n")
        md.append("```json\n" + json.dumps(outline_obj, ensure_ascii=False, indent=2) + "\n```\n\n")

        md.append("## Steps\n")
        for s in steps:
            md.append(f"- {s}\n")
        md.append("\n")

        md.append("## Step Explanations\n")
        for i, exp in enumerate(step_explanations):
            step_title = steps[i] if i < len(steps) else f"Step {i+1}"
            md.append(f"### {step_title}\n")
            md.append(exp + "\n\n")

        md.append("## Selected Materials\n")
        md.append("```json\n" + json.dumps(state.selected_material_ids, ensure_ascii=False, indent=2) + "\n```\n\n")

        md.append("## Retrieval Debug\n")
        md.append("```json\n" + json.dumps(retrieval_info, ensure_ascii=False, indent=2) + "\n```\n\n")

        raw_text = "".join(md)
        md_text = self.format_markdown_chunked(raw_text)
        md_text = self.strip_outer_code_fence(md_text)

        os.makedirs(self.settings.cache_dir, exist_ok=True)
        safe_id = state.current_event_id.replace(":", "_")
        md_path = os.path.join(self.settings.cache_dir, f"{safe_id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        pdf_path = os.path.join(self.settings.cache_dir, f"{safe_id}.pdf")
        ok = self.tools["render_markdown_to_pdf"](md_path, pdf_path)

        if ok and os.path.exists(pdf_path):
            state.debug["doc_writer"] = {"md_path": md_path, "pdf_path": pdf_path}
            log_msg(f"doc.created local_md={md_path} local_pdf={pdf_path}")
            self._request_feedback(state.course_id, state.current_event_id, "assignment", pdf_path, state)
        else:
            log_msg("doc.create.failed")

        return state

    def codegen_node(self, state: WFState) -> WFState:
        outline = state.debug.get("outline", {}) or {}
        steps = outline.get("steps", []) or []
        if not steps:
            state.debug["codegen"] = {"skipped": True, "reason": "no steps"}
            return state

        context = json.dumps(
            {
                "assignment_analyzer": state.debug.get("assignment_analyzer", {}),
                "outline": outline,
                "step_explanations": state.debug.get("step_explanations", []),
            },
            ensure_ascii=False,
        )

        step_snippets = []
        for s in steps:
            prompt = CODEGEN_STEP_PROMPT.format(step=s, context=context)
            res = self.agents["codegen_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
            raw = self._agent_last_text(res)
            obj = extract_json_block(raw) or {}
            if obj:
                step_snippets.append(obj)

        stitched_prompt = CODEGEN_STITCH_PROMPT.format(snippets=json.dumps(step_snippets, ensure_ascii=False)[:18000])
        res2 = self.agents["codegen_agent"].invoke({"messages": [{"role": "user", "content": stitched_prompt}]})
        raw2 = self._agent_last_text(res2)
        stitched = extract_json_block(raw2) or {}

        safe_id = state.current_event_id.replace(":", "_")
        os.makedirs(self.settings.cache_dir, exist_ok=True)
        scaffold_path = os.path.join(self.settings.cache_dir, f"{safe_id}_scaffold.py")
        notes_path = os.path.join(self.settings.cache_dir, f"{safe_id}_scaffold.md")

        content = stitched.get("content") if isinstance(stitched, dict) else None
        if not content:
            content = "# Scaffold generation failed. Please review step snippets."

        with open(scaffold_path, "w", encoding="utf-8") as f:
            f.write(content)

        with open(notes_path, "w", encoding="utf-8") as f:
            f.write("# Scaffold Notes\n\n")
            f.write(json.dumps({"step_snippets": step_snippets, "stitched": stitched}, ensure_ascii=False, indent=2))

        state.debug["codegen"] = {
            "scaffold_path": scaffold_path,
            "notes_path": notes_path,
            "steps": len(steps),
        }
        log_msg(f"codegen.done scaffold={scaffold_path}")
        return state

    def digest_node(self, state: WFState) -> WFState:
        since = now_ts() - int(self.settings.lookback_hours_for_digest * 3600)
        evs = self.db.get_recent_events(state.course_id, since)[:60]
        if not evs:
            state.task_type = "no_op"
            return state

        updates = "\n".join(
            [f"- [{e['event_type']}] {e.get('title','')}: {(e.get('text') or '')[:160]}" for e in evs]
        )
        prompt = DIGEST_PROMPT.format(updates=updates)

        result = self.agents["digest_agent"].invoke({"messages": [{"role": "user", "content": prompt}]})
        raw = self._agent_last_text(result)
        obj = extract_json_block(raw) or {}
        if nonempty(obj.get("digest")):
            self.tools["notify_telegram"].invoke({"message": "Digest:\n" + obj["digest"][:1200]})

        state.debug["digest"] = obj
        log_msg(f"digest.done highlights={(obj.get('highlights') or [])[:6]}")
        return state

    def advance_node(self, state: WFState) -> WFState:
        if state.current_event_id:
            state.processed_event_ids.append(state.current_event_id)
        if state.current_event_id in state.new_event_ids:
            state.new_event_ids = [e for e in state.new_event_ids if e != state.current_event_id]
        state.current_event_id = None
        state.task_type = ""
        return state

    def route_from_router(self, state: WFState) -> str:
        return state.task_type or "no_op"

    def route_after_syllabus(self, state: WFState) -> str:
        return "awaiting" if state.awaiting_user else "have_syllabus"

    def route_after_advance(self, state: WFState) -> str:
        return "more" if state.new_event_ids else "done"

    def route_after_summaries(self, state: WFState) -> str:
        return "quiz" if state.task_type == "quiz_flow" else "assignment"

    def build_graph(self):
        graph = StateGraph(WFState)
        graph.set_entry_point("poller")

        graph.add_node("poller", self.poller_node)
        graph.add_node("material_cache", self.material_cache_node)
        graph.add_node("router", self.router_node)

        graph.add_node("quiz_detector", self.quiz_detector_node)
        graph.add_node("syllabus_extractor", self.syllabus_extractor_node)
        graph.add_node("hitl_syllabus", self.human_input_syllabus_node)
        graph.add_node("awaiting", lambda s: s)

        graph.add_node("assignment_analyzer", self.assignment_analyzer_node)
        graph.add_node("retrieval", self.retrieval_node)
        graph.add_node("ensure_summaries", self.ensure_summaries_node)

        graph.add_node("quiz_topic_expand", self.quiz_topic_expand_node)
        graph.add_node("quiz_subtopic_explain", self.quiz_subtopic_explain_node)
        graph.add_node("quiz_qa", self.quiz_qa_node)
        graph.add_node("quiz_doc_writer", self.quiz_doc_writer_node)

        graph.add_node("outline", self.outline_node)
        graph.add_node("step_explain", self.step_explain_node)
        graph.add_node("doc_writer", self.doc_writer_node)
        graph.add_node("codegen", self.codegen_node)

        graph.add_node("digest", self.digest_node)
        graph.add_node("advance", self.advance_node)

        graph.add_edge("poller", "material_cache")
        graph.add_edge("material_cache", "router")

        graph.add_conditional_edges(
            "router",
            self.route_from_router,
            {
                "quiz_flow": "quiz_detector",
                "assignment_flow": "assignment_analyzer",
                "digest_flow": "digest",
                "no_op": "advance",
            },
        )

        graph.add_edge("quiz_detector", "syllabus_extractor")
        graph.add_conditional_edges(
            "syllabus_extractor",
            self.route_after_syllabus,
            {"awaiting": "awaiting", "have_syllabus": "retrieval"},
        )
        graph.add_edge("hitl_syllabus", "retrieval")
        graph.add_edge("retrieval", "ensure_summaries")

        graph.add_edge("assignment_analyzer", "retrieval")

        graph.add_conditional_edges(
            "ensure_summaries",
            self.route_after_summaries,
            {"quiz": "quiz_topic_expand", "assignment": "outline"},
        )

        graph.add_edge("quiz_topic_expand", "quiz_subtopic_explain")
        graph.add_edge("quiz_subtopic_explain", "quiz_qa")
        graph.add_edge("quiz_qa", "quiz_doc_writer")
        graph.add_edge("quiz_doc_writer", "advance")

        graph.add_edge("outline", "step_explain")
        graph.add_edge("step_explain", "doc_writer")
        graph.add_edge("doc_writer", "codegen")
        graph.add_edge("codegen", "advance")

        graph.add_edge("digest", "advance")
        graph.add_edge("awaiting", END)

        graph.add_conditional_edges(
            "advance",
            self.route_after_advance,
            {"more": "router", "done": END},
        )

        return graph.compile()
