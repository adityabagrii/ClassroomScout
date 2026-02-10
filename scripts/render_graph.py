import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.auth import get_creds, build_clients
from agents.config import Settings
from agents.db import DB
from agents.tools import build_tools
from agents.agents import build_agents
from agents.workflow import Workflow


def parse_args():
    p = argparse.ArgumentParser(description="Render LangGraph workflow to PNG")
    p.add_argument("--credentials", default="credentials.json")
    p.add_argument("--token", default="token.json")
    p.add_argument("--db-path", default="classroom_agent.sqlite")
    p.add_argument("--cache-dir", default="cache_files")
    p.add_argument("--model", default="nvidia/llama-3.1-nemotron-ultra-253b-v1")
    p.add_argument("--out", default="workflow.png")
    return p.parse_args()


def main():
    args = parse_args()

    settings = Settings(
        credentials_path=Path(args.credentials),
        token_path=Path(args.token),
        db_path=Path(args.db_path),
        cache_dir=Path(args.cache_dir),
        nemotron_model=args.model,
    )

    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    creds = get_creds(settings.credentials_path, settings.token_path)
    classroom, drive, docs = build_clients(creds)

    db = DB(str(settings.db_path))
    db.init()

    tools, allowed_mimes = build_tools(classroom, drive, docs, db, str(settings.cache_dir))
    agents = build_agents(settings.nemotron_model, tools)

    wf = Workflow(settings, db, tools, agents, allowed_mimes)
    app = wf.build_graph()
    graph = app.get_graph()

    png = graph.draw_mermaid_png()
    Path(args.out).write_bytes(png)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
