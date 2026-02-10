# ClassroomScout - Automating Your Study

ClassroomScout is a production-grade automation system for Google Classroom that continuously monitors all ACTIVE courses you’re enrolled in, detects important updates, and transforms raw classroom activity into structured, actionable study artifacts.

Instead of manually tracking announcements, quizzes, and assignments, ClassroomScout uses a modular multi-agent workflow to triage new events, retrieve relevant materials, generate deep quiz preparation notes, create critical practice questions, and scaffold assignment code—automatically and in the background.

Designed for reliability and extensibility, ClassroomScout persists long-term state in SQLite, caches artifacts locally, and exposes a CLI for fine-grained control over execution. The result is an always-running academic companion that reduces cognitive overhead and lets you focus on learning rather than logistics.

## What It Does

- Lists all ACTIVE courses and processes each course independently.
- Polls announcements, coursework, and course materials.
- Routes events into:
  - Quiz flow (topic expansion, subtopic explanations, critical QA)
  - Assignment flow (analysis, outline, step explanations, document + code scaffold)
  - Digest flow (summary of recent updates)
- Tracks already seen events to avoid duplicate processing.
- Runs continuously in the background on a fixed interval (default 3.5 hours).

## Flow Details

**Quiz Flow**
- Detects quiz events and extracts or requests the syllabus.
- Expands each topic into subtopics + learning objectives.
- Generates detailed subtopic explanations with examples and key points.
- Produces 15–20 conceptual questions and 5–10 numerical questions.
- Builds a PDF study pack and sends it to Telegram for review.
- HITL: reply with `ACCEPT` or `FEEDBACK: <notes>` to regenerate with your guidance.

**Assignment Flow**
- Analyzes requirements and deliverables.
- Builds an outline and step-by-step plan with guidance.
- Generates a detailed PDF and a starter code scaffold.
- Sends the PDF to Telegram for review.
- HITL: reply with `ACCEPT` or `FEEDBACK: <notes>` to regenerate from the outline step.

## Project Layout

- `main.py`
  CLI entry point for the workflow.
- `agents/`
  - `auth.py`: OAuth and API client creation
  - `config.py`: Settings dataclass
  - `db.py`: SQLite schemas and helpers
  - `tools.py`: API + file tools
  - `prompts.py`: Prompt templates for agents
  - `agents.py`: LLM agent setup
  - `workflow.py`: LangGraph workflow
  - `runner.py`: polling loop and course iteration

## Workflow Diagram

![Workflow Diagram](workflow.png)

## Installation

### 1) Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure Google API credentials

- Place `credentials.json` in the project root.
- The first run will open a browser for OAuth and create `token.json`.

### 4) Set-up NVIDIA NIM API Key
1. Create or sign in to your NVIDIA account.
2. Open the NVIDIA NIM portal and generate a new API key.
3. Copy the key and store it securely.

#### Set Your NVIDIA API Key
Export the key in your terminal so the CLI can read it:
```bash
export NVIDIA_API_KEY="YOUR_KEY_HERE"
```

### 5) Optional: Ensure PDF rendering support

PDF generation uses Pandoc + XeLaTeX. Install if you want PDFs.

- macOS:
```bash
brew install pandoc mactex
```

If Pandoc is missing, the workflow will still run but PDF generation may fail.

### 6) Configure Telegram (for updates + HITL)

Set environment variables:

```bash
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
export TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
```

## Usage

### List active courses

```bash
python main.py --list-courses
```

### Run once across all ACTIVE courses

```bash
python main.py --once
```

### Run continuously (default every 3.5 hours)

```bash
python main.py
```

### Run for specific course IDs

```bash
python main.py --once --course-ids "123456,987654"
```

### Run in the background

```bash
nohup python main.py > classroom.log 2>&1 &
```

## CLI Arguments

- `--credentials` path to `credentials.json`
- `--token` path to `token.json`
- `--db-path` path to SQLite DB
- `--cache-dir` cache directory for artifacts
- `--course-ids` comma-separated list of course IDs (optional)
- `--once` run a single pass and exit
- `--poll-hours` polling interval in hours
- `--allow-hitl` enable interactive input if syllabus is missing
- `--model` LLM model name
- `--list-courses` list active courses and exit
- `--telegram-bot-token` Telegram bot token (or use `TELEGRAM_BOT_TOKEN`)
- `--telegram-chat-id` Telegram chat ID (or use `TELEGRAM_CHAT_ID`)

## Output Artifacts

All generated artifacts are stored per course in `artifacts_<course_name>/`.

- Quiz flow:
  - `*_quiz.md`, `*_quiz.pdf`
- Assignment flow:
  - `*.md`, `*.pdf`
  - `*_scaffold.py` (initial code scaffold)
  - `*_scaffold.md` (codegen notes + step snippets)

## Workflow Details

### Polling & Deduplication

Each course uses a per‑course checkpoint key (e.g., `poll:last_ts:<course_id>`). New events are only processed once and stored in a per‑course SQLite DB named `<course_name>_db.sqlite`.

### Quiz Flow Improvements

- Topics are expanded into subtopics and learning objectives.
- Subtopics receive detailed explanations.
- QA generation now targets:
  - 15–20 conceptual questions
  - 5–10 numerical questions
  - non‑vague, critical questions across all topics

### Assignment Flow Improvements

- Assignment analysis → outline → step explanations → document
- **Codegen agent** runs at the end of the assignment pipeline:
  - Generates scaffolding per step
  - Stitches into an initial `scaffold.py`

## Notes

- The workflow uses `langchain-nvidia-ai-endpoints`. Ensure your environment is configured to access the model.
- Telegram support requires setting `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, or passing `--telegram-bot-token` and `--telegram-chat-id`.
- If you want to disable PDF generation entirely, remove Pandoc or stub `render_markdown_to_pdf` in `agents/tools.py`.

## Quick Start

```bash
pip install -r requirements.txt
python main.py --once
```
