"""
Myfxbook login + community sentiment (single entry point)

How to use (no shell args needed)
- Set the variables below:
  - EMAIL, PASSWORD: your Myfxbook credentials
  - Or set SESSION_ID directly if you already have a session token
  - Optionally set SAVE_SESSION_PATH and SAVE_OUT_PATH to save JSON files
- Then run this file from your IDE or with Python; it will print the session and data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import requests


LOGIN_URL = "https://www.myfxbook.com/api/login.json"
SENTIMENT_URL = "https://www.myfxbook.com/api/get-community-outlook.json"

# ---- User-configurable variables ----
EMAIL: str = "s.becker@bastilities.com"
PASSWORD: str = "M4uf!by%"
SESSION_ID: str = ""
TIMEOUT_S: float = 10.0
SAVE_SESSION_PATH: Optional[str] = None  # e.g., "api/cache/myfxbook_session.json"
SAVE_OUT_PATH: Optional[str] = None      # e.g., "api/cache/myfxbook_sentiment.json"


def login_myfxbook(email: str, password: str, timeout: float = 10.0) -> str:
    params = {"email": email, "password": password}
    resp = requests.get(LOGIN_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()
    if data.get("error"):
        raise RuntimeError(data.get("message") or "Login failed")
    session = data.get("session")
    if not session:
        raise RuntimeError("No session returned from Myfxbook")
    return session


def get_myfxbook_community_sentiment(session: str, timeout: float = 10.0) -> dict:
    params = {"session": session}
    resp = requests.get(SENTIMENT_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(data.get("message") or "Myfxbook API error")
    return data


def main() -> None:
    # Determine session: prefer provided SESSION_ID, else login with EMAIL/PASSWORD
    session: Optional[str] = SESSION_ID.strip() or None
    if not session:
        if not EMAIL or not PASSWORD:
            raise SystemExit("Set SESSION_ID or both EMAIL and PASSWORD at the top of the file.")
        session = login_myfxbook(EMAIL, PASSWORD, timeout=TIMEOUT_S)

    print(session)

    if SAVE_SESSION_PATH:
        out_session = Path(SAVE_SESSION_PATH)
        out_session.parent.mkdir(parents=True, exist_ok=True)
        with open(out_session, "w", encoding="utf-8") as f:
            json.dump({"session": session}, f)

    data = get_myfxbook_community_sentiment(session, timeout=TIMEOUT_S)
    if SAVE_OUT_PATH:
        out = Path(SAVE_OUT_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        print(json.dumps(data))


if __name__ == "__main__":
    main()
