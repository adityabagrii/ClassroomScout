import argparse
import re
from pathlib import Path

from agents.auth import get_creds, build_clients
from agents.config import Settings
from agents.db import DB
from agents.tools import build_tools
from agents.agents import build_agents
from agents.workflow import Workflow
from agents.runner import run_loop, run_once_all_courses_with_factory
from agents.logging import log_msg


def parse_args():
    p = argparse.ArgumentParser(description="Automating Classroom updates with multi-agent workflow")
    p.add_argument("--credentials", default="credentials.json", help="Path to credentials.json")
    p.add_argument("--token", default="token.json", help="Path to token.json")
    p.add_argument("--db-path", default="classroom_agent.sqlite", help="SQLite DB path")
    p.add_argument("--cache-dir", default="cache_files", help="Cache directory")
    p.add_argument("--course-ids", default="", help="Comma-separated course IDs to process. If empty, uses ACTIVE courses.")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    p.add_argument("--poll-hours", type=float, default=3.5, help="Polling interval in hours for loop mode")
    p.add_argument("--allow-hitl", action="store_true", help="Allow interactive input for missing syllabus")
    p.add_argument("--model", default="nvidia/llama-3.1-nemotron-ultra-253b-v1", help="LLM model name")
    p.add_argument("--list-courses", action="store_true", help="List ACTIVE courses and exit")
    return p.parse_args()


def list_active_courses(classroom) -> list:
    courses = []
    token = None
    while True:
        resp = classroom.courses().list(pageSize=100, pageToken=token, courseStates="ACTIVE").execute()
        courses.extend(resp.get("courses", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return courses


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9-_]+", "-", name.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "course"


def main():
    args = parse_args()

    settings = Settings(
        credentials_path=Path(args.credentials),
        token_path=Path(args.token),
        db_path=Path(args.db_path),
        cache_dir=Path(args.cache_dir),
        poll_hours=args.poll_hours,
        allow_hitl=args.allow_hitl,
        nemotron_model=args.model,
    )

    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    creds = get_creds(settings.credentials_path, settings.token_path)
    classroom, drive, docs = build_clients(creds)

    course_ids = [c.strip() for c in args.course_ids.split(",") if c.strip()]

    if args.list_courses:
        courses = list_active_courses(classroom)
        log_msg("Active courses:")
        for c in courses:
            log_msg(f"- {c.get('name','(no name)')} ({c.get('id')})")
        return

    courses = list_active_courses(classroom)
    if course_ids:
        selected = [c for c in courses if c.get("id") in course_ids]
        if not selected:
            log_msg("No matching courses found for provided --course-ids")
            return
    else:
        selected = courses

    log_msg("Active courses:")
    for c in selected:
        log_msg(f"- {c.get('name','(no name)')} ({c.get('id')})")

    def build_app_for_course(course):
        course_name = course.get("name") or "course"
        course_id = course.get("id") or "unknown"
        safe_name = slugify(course_name)
        course_cache = Path(f"artifacts_{safe_name}")
        course_cache.mkdir(parents=True, exist_ok=True)

        course_db_path = Path(f"{safe_name}_db.sqlite")
        db = DB(str(course_db_path))
        db.init()

        tools, allowed_mimes = build_tools(classroom, drive, docs, db, str(course_cache))
        agents = build_agents(settings.nemotron_model, tools)
        course_settings = Settings(
            credentials_path=settings.credentials_path,
            token_path=settings.token_path,
            db_path=course_db_path,
            cache_dir=course_cache,
            max_pages_or_slides=settings.max_pages_or_slides,
            max_summarize_per_run=settings.max_summarize_per_run,
            max_chars=settings.max_chars,
            overlap=settings.overlap,
            max_chunks=settings.max_chunks,
            max_completion_tokens=settings.max_completion_tokens,
            top_k_retrieval=settings.top_k_retrieval,
            lookback_hours_for_digest=settings.lookback_hours_for_digest,
            use_embeddings=settings.use_embeddings,
            embed_model_name=settings.embed_model_name,
            nemotron_model=settings.nemotron_model,
            poll_hours=settings.poll_hours,
            allow_hitl=settings.allow_hitl,
        )
        wf = Workflow(course_settings, db, tools, agents, allowed_mimes)
        app = wf.build_graph()
        return app

    if args.once:
        run_once_all_courses_with_factory(selected, build_app_for_course, course_ids if course_ids else None)
    else:
        run_loop(selected, build_app_for_course, settings.poll_hours, course_ids if course_ids else None)


if __name__ == "__main__":
    main()
