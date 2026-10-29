"""v23 -> v24: Krea 2 generate history moves to ComfyUI's cfg scale.

Krea 2 now takes ComfyUI's cfg everywhere (1 = no CFG). Generate tasks saved
before held Krea's own guidance (cfg - 1) in ``tasks.generate_params``; this
moves their ``cfg_scale`` and the numbers of a ``cfg_scale`` XY axis up by 1
and marks the snapshot ``schema_version`` 2, the same upgrade the frontend
applies to old PNG metadata (paramsSnapshot.upgradeSnapshot).

Data only, no schema change; snapshots already at version 2 are left alone.
"""
from __future__ import annotations

import json
import sqlite3


def _shift(value):
    try:
        return round(float(value) + 1.0, 6)
    except (TypeError, ValueError):
        return value


def _shift_raw(raw: str) -> str:
    parts = []
    for part in str(raw).split(","):
        text = part.strip()
        try:
            parts.append(f"{round(float(text) + 1.0, 6):g}" if text else part)
        except ValueError:
            parts.append(part)
    return ", ".join(parts)


def upgrade_params(params: dict) -> dict | None:
    """The upgraded snapshot, or None when it needs no change."""
    if params.get("model_family") != "krea2" or int(params.get("schema_version") or 1) >= 2:
        return None
    out = dict(params)
    if out.get("cfg_scale") is not None:
        out["cfg_scale"] = _shift(out["cfg_scale"])
    draft = out.get("xy_draft")
    if isinstance(draft, dict):
        draft = dict(draft)
        for key in ("x", "y"):
            axis = draft.get(key)
            if isinstance(axis, dict) and axis.get("axis") == "cfg_scale":
                draft[key] = {**axis, "raw": _shift_raw(axis.get("raw", ""))}
        out["xy_draft"] = draft
    out["schema_version"] = 2
    return out


def migrate(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "generate_params" not in columns:
        return
    rows = conn.execute(
        "SELECT id, generate_params FROM tasks WHERE generate_params LIKE '%krea2%'"
    ).fetchall()
    for task_id, raw in rows:
        try:
            params = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(params, dict):
            continue
        upgraded = upgrade_params(params)
        if upgraded is not None:
            conn.execute(
                "UPDATE tasks SET generate_params = ? WHERE id = ?",
                (json.dumps(upgraded, ensure_ascii=False), task_id),
            )
