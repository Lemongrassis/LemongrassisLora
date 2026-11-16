"""Epoch-level adaptive learning rate (plateau tracker).

Ported from Fizgig (Apache-2.0), ``src/fizgig/training/adaptive_lr.py``:
Copyright 2026 shootthesound and Fizgig contributors.
https://github.com/shootthesound/Fizgig

The run starts at the geometric midpoint of ``min_lr`` and ``max_lr`` (the
Learning Rate field is ignored). At each epoch end, on the mean loss of the
last 3 epochs:

- loss improved 2 epochs in a row -> rate x1.25 (up to ``max_lr``)
- loss plateau -> rate x0.5 (down to ``min_lr``)
- adapter weight norm grew by more than 30% in one epoch -> rate x0.5, the
  weights are blended 70/30 toward the previous epoch's snapshot and the
  optimizer state is restored

A cut made within 3 epochs of a probe up marks that rate as too high: later
probes stop one step below it.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AdaptiveLR:
    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30
    CEILING_EPOCHS = 3
    SMOOTH_EPOCHS = 3

    def __init__(self, min_lr: float, max_lr: float):
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        if not (0 < self.min_lr <= self.max_lr):
            raise ValueError("adaptive_lr needs 0 < adaptive_lr_min <= adaptive_lr_max")
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.recent_losses: list[float] = []
        self.ceiling = None
        self.probe_epoch = None
        self.probe_lr = None
        self.snapshot = None  # in memory only: the first epoch after a resume cannot roll back

    @property
    def start_lr(self) -> float:
        return math.sqrt(self.min_lr * self.max_lr)

    def state_dict(self) -> dict:
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm, "recent_losses": list(self.recent_losses),
                "ceiling": self.ceiling, "probe_epoch": self.probe_epoch, "probe_lr": self.probe_lr}

    def load_state_dict(self, d: dict) -> None:
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")
        self.recent_losses = [float(x) for x in (d.get("recent_losses") or [])][-self.SMOOTH_EPOCHS:]
        self.ceiling = d.get("ceiling")
        self.probe_epoch = d.get("probe_epoch")
        self.probe_lr = d.get("probe_lr")

    def _top(self) -> float:
        return self.max_lr if self.ceiling is None else min(self.max_lr, self.ceiling)

    def _learn_ceiling(self, epoch: int, cur_lr: float) -> str:
        if (self.probe_epoch is None or self.probe_lr is None
                or epoch - self.probe_epoch > self.CEILING_EPOCHS
                or cur_lr < self.probe_lr * 0.999):
            return ""
        cap = max(self.min_lr, self.probe_lr / 1.25)
        self.probe_epoch = self.probe_lr = None
        if self.ceiling is not None and self.ceiling <= cap:
            return ""
        self.ceiling = cap
        return f"; ceiling {cap:.2e} learned"

    @staticmethod
    def _weight_norm(params) -> float:
        wn = 0.0
        with torch.no_grad():
            for p in params:
                wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, params, optimizer) -> None:
        with torch.no_grad():
            weights = [p.detach().clone().to("cpu") for p in params]

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:  # noqa: BLE001
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, params, optimizer) -> None:
        with torch.no_grad():
            for p, prev in zip(params, self.snapshot["weights"]):
                p.copy_(self.BLEND * prev.to(device=p.device, dtype=p.dtype) + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            lrs = [g["lr"] for g in optimizer.param_groups]
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:  # noqa: BLE001
                pass
            for g, lr in zip(optimizer.param_groups, lrs):
                g["lr"] = lr

    def epoch_boundary(self, epoch: int, current_loss: float, params, optimizer) -> str:
        """``epoch`` is 0-based: epoch 0 arms the baseline, later ones adjust the rate.
        Returns the log line."""
        params = [p for p in params if p.requires_grad]
        raw_loss = float(current_loss)
        self.recent_losses = (self.recent_losses + [raw_loss])[-self.SMOOTH_EPOCHS:]
        current_loss = sum(self.recent_losses) / len(self.recent_losses)
        cur_lr = optimizer.param_groups[0]["lr"]
        if self.best_loss is None and self.prev_weight_norm is None:
            self.best_loss = current_loss
            self.prev_weight_norm = self._weight_norm(params)
            self._snapshot(params, optimizer)
            return f"[adaptive_lr] epoch {epoch + 1}: loss={raw_loss:.4f} lr={cur_lr:.2e} | ARMED"

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        new_lr = cur_lr
        cur_wn = self._weight_norm(params)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        if weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = (f"weight norm {weight_growth * 100:+.0f}% > "
                                f"{self.WEIGHT_GROWTH_THRESHOLD * 100:.0f}%")

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * 0.5, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(params, optimizer)
                    note = "; weights blended 70/30 toward the last epoch, optimizer restored"
                if candidate < cur_lr:
                    note += self._learn_ceiling(epoch, cur_lr)
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * 1.25, self._top())
                if candidate > cur_lr * 1.001:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                    self.probe_epoch, self.probe_lr = epoch, candidate
                else:
                    action = "HOLD (capped)"
                    reason = "loss improving, at the top rate"
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * 0.5, self.min_lr)
                if candidate < cur_lr:
                    reason = f"loss plateau, streak {self.bad_streak}" + self._learn_ceiling(epoch, cur_lr)
                    new_lr = candidate
                    action = "REDUCE"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            scale = new_lr / cur_lr
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * scale
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth * 100:+.0f}%" if weight_growth is not None else "-"
        self.prev_weight_norm = cur_wn
        self._snapshot(params, optimizer)
        return (f"[adaptive_lr] epoch {epoch + 1}: loss={raw_loss:.4f} (avg {current_loss:.4f}) "
                f"lr={lr_str} weight norm {wn_str} | {action} ({reason})")


def build_adaptive_lr(args: Any, optimizer, scheduler) -> AdaptiveLR | None:
    """Build the tracker and set the start rate; None when ``adaptive_lr`` is off."""
    if not bool(getattr(args, "adaptive_lr", False)):
        return None
    if scheduler is not None:
        raise ValueError(
            "adaptive_lr owns the learning rate: set lr_scheduler to none "
            "(and pick an optimizer that does not set its own rate)"
        )
    adaptive = AdaptiveLR(
        float(getattr(args, "adaptive_lr_min", 1e-4) or 1e-4),
        float(getattr(args, "adaptive_lr_max", 4e-4) or 4e-4),
    )
    base = optimizer.param_groups[0]["lr"] or adaptive.start_lr
    scale = adaptive.start_lr / base
    for pg in optimizer.param_groups:
        pg["lr"] = pg["lr"] * scale if pg["lr"] else adaptive.start_lr
    logger.info(
        "[adaptive_lr] on: start %.3e, min %.3e, max %.3e (learning_rate is ignored)",
        adaptive.start_lr, adaptive.min_lr, adaptive.max_lr,
    )
    return adaptive
