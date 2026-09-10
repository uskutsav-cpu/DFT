"""Transactional local job ledger with explicit retry and verified result cache.

Running jobs are never silently stolen. A crashed worker requires an explicit
mark_failed/retry after the operator checks that the process has stopped.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from .provenance import canonical_json, digest, file_digest, utc_now


class JobStore:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
            key TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, claim_token TEXT,
            updated TEXT NOT NULL, result_path TEXT, result_sha256 TEXT, error TEXT
        )""")

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def register(self, payload: dict) -> str:
        key = digest(payload)
        self.connection.execute(
            "INSERT OR IGNORE INTO jobs(key,payload,status,updated) VALUES(?,?,?,?)",
            (key, canonical_json(payload), "pending", utc_now()),
        )
        return key

    def claim(self, key: str) -> str | None:
        token = uuid.uuid4().hex
        cursor = self.connection.execute(
            "UPDATE jobs SET status='running', attempts=attempts+1, claim_token=?, updated=? "
            "WHERE key=? AND status='pending'",
            (token, utc_now(), key),
        )
        return token if cursor.rowcount == 1 else None

    def finish(
        self, key: str, token: str, result_path: str | Path, *, success: bool, error: str = ""
    ):
        path = Path(result_path).resolve()
        sha = file_digest(path)
        cursor = self.connection.execute(
            "UPDATE jobs SET status=?,result_path=?,result_sha256=?,error=?,updated=?,claim_token=NULL "
            "WHERE key=? AND status='running' AND claim_token=?",
            ("succeeded" if success else "failed", str(path), sha, error, utc_now(), key, token),
        )
        if cursor.rowcount != 1:
            raise ValueError("invalid or stale job claim")

    def mark_failed(self, key: str, token: str, error: str):
        cursor = self.connection.execute(
            "UPDATE jobs SET status='failed',error=?,updated=?,claim_token=NULL "
            "WHERE key=? AND status='running' AND claim_token=?",
            (error, utc_now(), key, token),
        )
        if cursor.rowcount != 1:
            raise ValueError("invalid or stale job claim")

    def retry(self, key: str, *, max_attempts: int = 3) -> bool:
        from .validation import integer

        integer(max_attempts, "max_attempts", 1)
        cursor = self.connection.execute(
            "UPDATE jobs SET status='pending',updated=? WHERE key=? AND status='failed' AND attempts<?",
            (utc_now(), key, max_attempts),
        )
        return cursor.rowcount == 1

    def get(self, key: str) -> dict:
        row = self.connection.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    def cached_result(self, key: str) -> Path | None:
        row = self.get(key)
        if row["status"] != "succeeded":
            return None
        path = Path(row["result_path"])
        if not path.is_file() or file_digest(path) != row["result_sha256"]:
            raise ValueError("cached result missing or changed; refusing silent reuse")
        return path

    def counts(self) -> dict:
        return {
            row["status"]: row["n"]
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS n FROM jobs GROUP BY status"
            )
        }
