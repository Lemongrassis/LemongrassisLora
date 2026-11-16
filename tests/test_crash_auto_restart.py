"""A crashed training run is put back in the queue from its latest recovery point.

Supervisor._finish_slot -> _maybe_auto_restart: restart conditions (train task, reached
the training loop, streak under crash_auto_restarts), the resume point it hands to the
paused_* pipeline, the streak reset on progress, and the appended run.log.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio import db
from studio.supervisor import Supervisor, _Slot
import studio.supervisor.core as core


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "configs").mkdir()
    log_dir = tmp_path / "tasks"
    monkeypatch.setattr(core, "task_log_path", lambda tid: log_dir / str(tid) / "run.log")
    snapshots: dict[int, dict[str, Any]] = {}
    from studio.services import task_snapshot
    monkeypatch.setattr(
        task_snapshot, "read_snapshot_config",
        lambda tid: {"yaml": "", "config": snapshots[tid]} if tid in snapshots else None,
    )
    events: list[dict[str, Any]] = []
    sup = Supervisor(
        on_event=events.append,
        cmd_builder=lambda *_: ["echo"],
        db_path=db_path,
        logs_dir=tmp_path / "logs",
        configs_dir=tmp_path / "configs",
        poll_interval=10,
    )
    return {"db": db_path, "root": tmp_path, "sup": sup, "events": events,
            "snapshots": snapshots, "log_dir": log_dir}


def _create_task(env, **fields: Any) -> int:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(conn, tid, status="running", **fields)
    return tid


def _task(env, tid: int) -> dict[str, Any]:
    with db.connection_for(env["db"]) as conn:
        return db.get_task(conn, tid)


def _recovery_point(env, tid: int) -> tuple[Path, Path]:
    d = env["root"] / "state" / str(tid)
    d.mkdir(parents=True, exist_ok=True)
    pt, cfg = d / "auto_epoch_state.pt", d / "auto_epoch_state.config.json"
    pt.write_bytes(b"x")
    cfg.write_text("{}", encoding="utf-8")
    return pt, cfg


def _crash(env, tid: int, *, loop_started: bool = True, rc: int = 1) -> None:
    log = env["log_dir"] / str(tid) / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fp:
        fp.write("step 10\nTraceback (most recent call last):\nRuntimeError: CUDA error\n")
    slot = _Slot(name="train")
    slot.kind, slot.id, slot.proc = "task", tid, MagicMock()
    slot.train_loop_started = loop_started
    env["sup"]._finish_slot(slot, rc)


def test_crash_after_recovery_point_requeues_from_it(env) -> None:
    tid = _create_task(env, priority=0)
    with db.connection_for(env["db"]) as conn:
        other = db.create_task(conn, name="other", config_name="c", priority=5)
    pt, cfg = _recovery_point(env, tid)
    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, tid, last_state_path=str(pt), last_config_path=str(cfg),
                       last_state_epoch=2, last_state_step=200)

    _crash(env, tid)

    task = _task(env, tid)
    assert task["status"] == "pending"
    assert task["paused_state_path"] == str(pt)
    assert task["paused_config_path"] == str(cfg)
    assert task["paused_step"] == 200
    assert task["auto_restarts"] == 1 and task["auto_restart_streak"] == 1
    assert "RuntimeError: CUDA error" in task["last_crash_msg"]
    assert task["priority"] > _task(env, other)["priority"]  # restarts ahead of the queue
    log = (env["log_dir"] / str(tid) / "run.log").read_text(encoding="utf-8")
    assert "[auto-restart]" in log and "step 200" in log
    types = [e["type"] for e in env["events"]]
    assert "task_auto_restarted" in types
    assert {"type": "task_state_changed", "task_id": tid, "status": "pending"} in env["events"]


def test_crash_without_recovery_point_starts_over(env) -> None:
    tid = _create_task(env, last_state_path="/gone/auto_epoch_state.pt", last_state_step=500)
    _crash(env, tid)
    task = _task(env, tid)
    assert task["status"] == "pending"
    assert task["paused_state_path"] is None
    assert task["last_state_path"] is None and task["last_state_step"] is None


def test_crash_before_training_loop_fails(env) -> None:
    tid = _create_task(env)
    _crash(env, tid, loop_started=False)
    task = _task(env, tid)
    assert task["status"] == "failed"
    assert not task["auto_restarts"]


def test_non_train_task_is_not_restarted(env) -> None:
    tid = _create_task(env, task_type="reg_ai")
    _crash(env, tid)
    assert _task(env, tid)["status"] == "failed"


def test_limit_from_config_snapshot(env) -> None:
    tid = _create_task(env)
    env["snapshots"][tid] = {"crash_auto_restarts": 2}
    for expected in ("pending", "pending", "failed"):
        _crash(env, tid)
        assert _task(env, tid)["status"] == expected
        if expected == "pending":
            with db.connection_for(env["db"]) as conn:
                db.update_task(conn, tid, status="running")
    assert _task(env, tid)["auto_restarts"] == 2


def test_zero_disables_restarts(env) -> None:
    tid = _create_task(env)
    env["snapshots"][tid] = {"crash_auto_restarts": 0}
    _crash(env, tid)
    assert _task(env, tid)["status"] == "failed"


def test_default_is_one_restart_then_failed(env) -> None:
    """Default crash_auto_restarts = 1: a second crash stays failed, even after progress."""
    tid = _create_task(env)
    _crash(env, tid)
    assert _task(env, tid)["status"] == "pending"
    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, tid, status="running")
    slot = _Slot(name="train")
    cb = env["sup"]._make_task_log_callback(slot, tid)
    cb("__EVENT__:auto_epoch_backup_written:" + json.dumps({
        "state_path": "/s/auto_epoch_state.pt", "config_path": "/s/c.json",
        "epoch": 2, "step": 260,
    }))
    _crash(env, tid)
    task = _task(env, tid)
    assert task["status"] == "failed"
    assert task["auto_restarts"] == 1


def test_manual_resume_gives_a_fresh_restart(env, monkeypatch) -> None:
    from studio.api.routers.queue import lifecycle

    monkeypatch.setattr(db, "STUDIO_DB", env["db"])
    tid = _create_task(env)
    pt, cfg = _recovery_point(env, tid)
    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, tid, status="failed", auto_restarts=1, auto_restart_streak=1,
                       last_state_path=str(pt), last_config_path=str(cfg), last_state_step=300)
    lifecycle.resume_task(tid)
    task = _task(env, tid)
    assert task["status"] == "pending" and task["auto_restart_streak"] == 0
    assert task["paused_state_path"] == str(pt)


def test_cancel_is_not_restarted(env) -> None:
    tid = _create_task(env)
    slot = _Slot(name="train")
    slot.kind, slot.id, slot.proc = "task", tid, MagicMock()
    slot.train_loop_started = True
    slot.cancel_pending = True
    env["sup"]._finish_slot(slot, 1)
    assert _task(env, tid)["status"] == "canceled"


def test_restart_spawn_appends_log_and_skips_old_events(env, monkeypatch) -> None:
    """The restarted run appends to run.log; the tailer starts past the old content."""
    tid = _create_task(env, auto_restart_streak=1)
    log = env["log_dir"] / str(tid) / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    old = b"__EVENT__:pause_state:{\"state_path\": \"/old.pt\"}\nold line\n"
    log.write_bytes(old)
    cfg = env["root"] / "configs" / "c.yaml"
    cfg.write_text("epochs: 1\n", encoding="utf-8")

    offsets: list[int] = []

    class _Tailer:
        def __init__(self, path, cb, *, start_offset=0):
            offsets.append(start_offset)

        def start(self):
            pass

    monkeypatch.setattr(core, "LogTailer", _Tailer)
    monkeypatch.setattr(core, "MonitorStatePoller", MagicMock())
    monkeypatch.setattr(core, "_resolve_monitor_state_path", lambda t: env["root"] / "m.json")
    sup = env["sup"]
    monkeypatch.setattr(sup, "_freeze_task_snapshot", lambda *a: None)
    monkeypatch.setattr(sup, "_maybe_split_validation", lambda *a: None)
    monkeypatch.setattr(sup, "_popen", lambda cmd, fp, extra_env=None: MagicMock(pid=1))
    slot = _Slot(name="train")
    sup._spawn_task(slot, _task(env, tid))
    slot.log_fp.close()

    assert offsets == [len(old)]
    assert log.read_bytes() == old  # kept, not truncated
