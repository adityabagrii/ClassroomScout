import os
import warnings
from pathlib import Path
from typing import Tuple

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from .logging import log_msg

SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.announcements.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.me.readonly",
    "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents",
]

# allow oauthlib to accept subset scopes without raising
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"


def get_creds(credentials_path: Path, token_path: Path) -> Credentials:
    log_msg("Starting OAuth credential loading...")
    log_msg(f"Credentials: {credentials_path} (exists={credentials_path.exists()})")
    log_msg(f"Token:       {token_path} (exists={token_path.exists()})")

    if not credentials_path.exists():
        raise FileNotFoundError(f"Missing {credentials_path}. Put credentials.json there.")

    if token_path.exists():
        log_msg("Loading existing token.json...")
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        log_msg(f"Loaded token. valid={creds.valid}, expired={creds.expired}, has_refresh={bool(creds.refresh_token)}")
        if creds.expired and creds.refresh_token:
            log_msg("Refreshing token...")
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
            log_msg("Token refreshed + saved.")
        log_msg(f"Granted scopes (token): {creds.scopes}")
        return creds

    log_msg("No token found -> launching OAuth flow...")
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Scope has changed.*")
        creds = flow.run_local_server(port=0, open_browser=True, prompt="consent")

    token_path.write_text(creds.to_json())
    log_msg(f"token.json saved at: {token_path}")
    log_msg(f"Granted scopes (auth): {creds.scopes}")

    req, got = set(SCOPES), set(creds.scopes or [])
    if got and req != got:
        log_msg("Scope mismatch (not fatal):")
        log_msg(f"  Requested-only: {sorted(req - got)}")
        log_msg(f"  Granted-only:   {sorted(got - req)}")

    return creds


def build_clients(creds: Credentials):
    log_msg("Building API clients...")
    classroom = build("classroom", "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)
    docs = build("docs", "v1", credentials=creds)
    log_msg("Clients built.")
    return classroom, drive, docs
