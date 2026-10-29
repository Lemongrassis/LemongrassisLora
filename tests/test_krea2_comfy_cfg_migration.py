"""v24: Krea 2 generate history moves from Krea's guidance to ComfyUI's cfg."""

from __future__ import annotations

import json
import sqlite3

from studio.infrastructure.migrations._v24_krea2_comfy_cfg import migrate, upgrade_params


def test_upgrade_moves_cfg_and_cfg_axis_once():
    old = {
        "schema_version": 1, "model_family": "krea2", "cfg_scale": 1,
        "xy_draft": {"x": {"axis": "cfg_scale", "raw": "0, 3.5", "loraIndex": None},
                     "y": {"axis": "steps", "raw": "8, 20", "loraIndex": None}},
    }
    new = upgrade_params(old)
    assert new["cfg_scale"] == 2.0
    assert new["xy_draft"]["x"]["raw"] == "1, 4.5"
    assert new["xy_draft"]["y"]["raw"] == "8, 20"
    assert new["schema_version"] == 2
    assert upgrade_params(new) is None
    assert upgrade_params({"schema_version": 1, "model_family": "anima", "cfg_scale": 4}) is None


def test_migrate_rewrites_only_old_krea2_tasks():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, generate_params TEXT)")
    rows = {
        1: {"schema_version": 1, "model_family": "krea2", "cfg_scale": 4.5},
        2: {"schema_version": 2, "model_family": "krea2", "cfg_scale": 1},
        3: {"schema_version": 1, "model_family": "anima", "cfg_scale": 4},
    }
    for task_id, params in rows.items():
        conn.execute("INSERT INTO tasks VALUES (?, ?)", (task_id, json.dumps(params)))
    migrate(conn)
    got = {i: json.loads(p)["cfg_scale"] for i, p in conn.execute("SELECT id, generate_params FROM tasks")}
    assert got == {1: 5.5, 2: 1, 3: 4}
