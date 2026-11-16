"""v22 -> v23: automatic restart of a training task after a crash.

New columns (tasks):

- `auto_restarts`: how many times the supervisor restarted this task after a crash
  (total, shown in the queue)
- `auto_restart_streak`: restarts since the task was last started by hand (the manual
  resume endpoint resets it), capped by the task config's `crash_auto_restarts`
- `last_crash_msg`: exit code + log tail of the latest crash that triggered a restart

All NULLABLE; existing rows need no backfill (NULL reads as 0 / no crash).
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "auto_restarts", "auto_restarts INTEGER")
    _add_column_if_missing(conn, "tasks", "auto_restart_streak", "auto_restart_streak INTEGER")
    _add_column_if_missing(conn, "tasks", "last_crash_msg", "last_crash_msg TEXT")
