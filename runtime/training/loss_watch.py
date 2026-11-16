"""Per-image loss watch and per-image learning rate.

Ported from Fizgig (Apache-2.0), ``src/fizgig/training/loss_logger.py``
``PerImageLossWatch`` -- thresholds, verdicts and multipliers are kept as is:
Copyright 2026 shootthesound and Fizgig contributors.
https://github.com/shootthesound/Fizgig

What is left out: auto-recaption, exclusion of images and the look-filter
warm-up (they need Fizgig's captioner and its dataset tools).

Every training sample's raw MSE is recorded with its image and its noise
level. A step's loss depends mostly on the drawn noise level, so an image is
judged by its residual: the loss minus the run's mean loss in the same noise
bucket. At each epoch end every image gets a verdict, and with per-image LR
on, a multiplier applied to its loss (with AdamW that is a learning rate
relative to the other images):

    easy       x1.1   lower third by difficulty, from epoch 3
    mid        x1.0
    learning   x1.0   hard but still descending
    watch      x1.0   looks stuck this epoch, not confirmed
    suspect    x0.7   an outlier by magnitude, before a trend exists
    stuck      x0.5   hard and not improving; halves every 2 epochs, floor x0.25
    exhausted  x0.6   proved a good run, then plateaued

Records are kept as sums per (image, epoch, noise bucket), which gives exactly
the residuals a flat record list would, in bounded memory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

N_BUCKETS = 40


def _bucket(t: float) -> int:
    return min(int(min(max(float(t), 0.0), 0.999999) * N_BUCKETS), N_BUCKETS - 1)


class PerImageLossWatch:
    def __init__(
        self,
        output_dir: str | os.PathLike | None,
        *,
        apply_lr: bool = False,
        warmup_epochs: int = 2,
        window: int = 5,
        throttle_mult: float = 0.5,
        easy_mult: float = 1.1,
        hi_q: float = 0.66,
        lo_q: float = 0.33,
        persist_on: int = 2,
        persist_off: int = 3,
        improve_frac: float = 0.12,
        improve_floor: float = 0.02,
        suspect_mult: float = 0.7,
        easy_from_epoch: int = 3,
        exhausted_mult: float = 0.6,
        exhaust_drop_frac: float = 0.3,
        exhaust_on: int = 2,
        stuck_floor: float = 0.25,
        escalate_every: int = 2,
        plateau_patience: int = 2,
        plateau_min_epochs: int = 8,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.apply_lr = bool(apply_lr)
        self.warmup_epochs = warmup_epochs
        self.window = window
        self.throttle_mult = throttle_mult
        self.easy_mult = easy_mult
        self.hi_q = hi_q
        self.lo_q = lo_q
        self.persist_on = persist_on
        self.persist_off = persist_off
        self.improve_frac = improve_frac
        self.improve_floor = improve_floor
        self.trend_window = 8
        self.suspect_mult = suspect_mult
        self.easy_from_epoch = easy_from_epoch
        self.exhausted_mult = exhausted_mult
        self.exhaust_drop_frac = exhaust_drop_frac
        self.exhaust_on = exhaust_on
        self.stuck_floor = stuck_floor
        self.escalate_every = escalate_every
        self.plateau_patience = plateau_patience
        self.plateau_min_epochs = plateau_min_epochs

        # key -> epoch -> bucket -> [loss sum, count]
        self._rec: dict[str, dict[int, dict[int, list[float]]]] = {}
        self._bsum = [0.0] * N_BUCKETS
        self._bcnt = [0] * N_BUCKETS
        self._epochs_seen: set[int] = set()
        self._mult: dict[str, float] = {}
        self.verdicts: dict[str, str] = {}
        self._stuck_votes: dict[str, int] = {}
        self._clear_votes: dict[str, int] = {}
        self._suspect_votes: dict[str, int] = {}
        self._exhaust_votes: dict[str, int] = {}
        self._stuck_epochs: dict[str, int] = {}
        self._confirmed_stuck: set[str] = set()
        self._last_improving_epoch: dict[str, int] = {}
        self._improving_streak: dict[str, int] = {}
        self._no_improve_streak = 0
        self._plateau_reported = False
        self.plateaued = False
        self.best_epoch_estimate: int | None = None

    # ---- per step ------------------------------------------------------

    def multiplier(self, key: str) -> float:
        """Loss multiplier for this image (changes only at epoch ends)."""
        if not self.apply_lr:
            return 1.0
        return self._mult.get(str(key), 1.0)

    def observe(self, key: str, epoch: int, timestep: float, loss: float) -> None:
        """Record one sample's raw (unscaled) loss. ``epoch`` is 1-based."""
        loss = float(loss)
        if loss != loss:  # NaN
            return
        b = _bucket(timestep)
        cell = self._rec.setdefault(str(key), {}).setdefault(int(epoch), {}).setdefault(b, [0.0, 0])
        cell[0] += loss
        cell[1] += 1
        self._bsum[b] += loss
        self._bcnt[b] += 1
        self._epochs_seen.add(int(epoch))

    # ---- per epoch -----------------------------------------------------

    @staticmethod
    def _slope(xs, ys) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

    def epoch_boundary(self, epoch: int) -> dict[str, str]:
        """Reclassify every image and refresh the multipliers. ``epoch`` is 1-based."""
        if not self._rec or len(self._epochs_seen) < self.warmup_epochs:
            return {}
        bucket_mean = [(self._bsum[i] / self._bcnt[i]) if self._bcnt[i] else 0.0 for i in range(N_BUCKETS)]

        stats: dict[str, dict[str, Any]] = {}
        for key, ep_map in self._rec.items():
            all_eps = sorted(ep_map)
            series = []
            raw_sum = 0.0
            raw_cnt = 0
            for e in all_eps:
                res_sum = 0.0
                cnt = 0
                for b, (s, c) in ep_map[e].items():
                    res_sum += s - c * bucket_mean[b]
                    cnt += c
                    raw_sum += s
                series.append(res_sum / cnt)
                raw_cnt += cnt

            tw = min(self.trend_window, len(series) - len(series) % 2)
            old_mean = new_mean = se = 0.0
            if tw >= 4:
                win = series[-tw:]
                half = tw // 2
                old_half, new_half = win[:half], win[half:]
                old_mean = sum(old_half) / half
                new_mean = sum(new_half) / half
                var = (sum((v - old_mean) ** 2 for v in old_half)
                       + sum((v - new_mean) ** 2 for v in new_half)) / max(tw - 2, 1)
                se = (2.0 * var / half) ** 0.5

            res = series[-self.window:]
            eps = all_eps[-self.window:]
            nb = min(3, len(series))
            baseline = sum(series[:nb]) / nb
            stats[key] = {
                "mean_residual": sum(res) / len(res),
                "slope": self._slope([float(e) for e in eps], res),
                "first": old_mean, "last": new_mean, "se": se,
                "trend_epochs": tw,
                "baseline": baseline,
                "total_drop": baseline - new_mean if tw >= 4 else 0.0,
                "mean_loss": raw_sum / raw_cnt,
                "epochs": len(all_eps),
            }

        ordered = sorted(s["mean_residual"] for s in stats.values())
        n = len(ordered)
        hi = ordered[int(self.hi_q * (n - 1))]
        lo = ordered[int(self.lo_q * (n - 1))]
        med = ordered[n // 2]
        iqr = ordered[int(0.75 * (n - 1))] - ordered[int(0.25 * (n - 1))]
        ext_hi = max(med + 1.5 * iqr, hi)
        ext_vhi = max(med + 3.0 * iqr, hi)
        n_ep = len(self._epochs_seen)

        new_mult: dict[str, float] = {}
        improving_count = 0
        for key, s in stats.items():
            drop = s["first"] - s["last"]
            improving = (s["trend_epochs"] >= 4
                         and drop >= max(self.improve_frac * max(s["first"], 0.0),
                                         self.improve_floor, s["se"]))
            s["improving"] = improving
            self._improving_streak[key] = self._improving_streak.get(key, 0) + 1 if improving else 0
            if self._improving_streak[key] >= 2:
                improving_count += 1
                self._last_improving_epoch[key] = max(1, epoch - self.trend_window // 2)
            good_run = (s["trend_epochs"] >= 4 and s["baseline"] > 0.0
                        and s["total_drop"] >= max(self.exhaust_drop_frac * s["baseline"],
                                                   2.0 * self.improve_floor))
            votes_stuck = (s["trend_epochs"] >= 4 and s["mean_residual"] >= ext_hi
                           and s["mean_residual"] > s["se"]
                           and s["last"] > 0.0 and not improving and not good_run)
            if key in self._confirmed_stuck:
                self._stuck_epochs[key] = self._stuck_epochs.get(key, 0) + 1
                self._clear_votes[key] = 0 if votes_stuck else self._clear_votes.get(key, 0) + 1
                if self._clear_votes[key] >= self.persist_off:
                    self._confirmed_stuck.discard(key)
                    self._clear_votes[key] = 0
                    self._stuck_votes[key] = 0
                    self._stuck_epochs[key] = 0
            else:
                self._stuck_votes[key] = self._stuck_votes.get(key, 0) + 1 if votes_stuck else 0
                if self._stuck_votes[key] >= self.persist_on:
                    self._confirmed_stuck.add(key)
                    self._clear_votes[key] = 0
                    self._stuck_epochs[key] = self._stuck_epochs.get(key, 0) + 1

            extreme = (s["mean_residual"] >= ext_hi and not improving and not good_run
                       and (s["trend_epochs"] < 4 or s["mean_residual"] > s["se"]))
            self._suspect_votes[key] = self._suspect_votes.get(key, 0) + 1 if extreme else 0
            suspect = (key not in self._confirmed_stuck
                       and (self._suspect_votes[key] >= 2
                            or (self._suspect_votes[key] >= 1 and s["mean_residual"] >= ext_vhi)))

            exhaust_now = good_run and not improving and s["last"] > 0.0
            self._exhaust_votes[key] = self._exhaust_votes.get(key, 0) + 1 if exhaust_now else 0
            exhausted = self._exhaust_votes[key] >= self.exhaust_on

            if key in self._confirmed_stuck:
                tenure = self._stuck_epochs.get(key, 1)
                mult = max(self.stuck_floor,
                           self.throttle_mult * (0.5 ** ((tenure - 1) // self.escalate_every)))
                verdict = "stuck"
                s["stuck_epochs"] = tenure
            elif suspect:
                verdict, mult = "suspect", self.suspect_mult
            elif exhausted:
                verdict, mult = "exhausted", self.exhausted_mult
            elif votes_stuck:
                verdict, mult = "watch", 1.0
            elif s["mean_residual"] >= hi:
                verdict, mult = "learning", 1.0
            elif s["mean_residual"] <= lo and n_ep >= self.easy_from_epoch:
                verdict, mult = "easy", self.easy_mult
            else:
                verdict, mult = "mid", 1.0
            s["verdict"] = verdict
            s["multiplier"] = mult if self.apply_lr else 1.0
            new_mult[key] = mult
        self._mult = new_mult
        self.verdicts = {k: s["verdict"] for k, s in stats.items()}

        # "Learning finished" estimate: nothing has been improving for a while;
        # the suggested checkpoint is the 75th percentile of per-image finish epochs.
        tol = max(1, round(0.05 * len(stats)))
        if n_ep >= self.plateau_min_epochs and improving_count <= tol:
            self._no_improve_streak += 1
        else:
            self._no_improve_streak = 0
        self.plateaued = self._no_improve_streak >= self.plateau_patience
        finishes = sorted(self._last_improving_epoch.values())
        if finishes:
            self.best_epoch_estimate = int(finishes[int(0.75 * (len(finishes) - 1))])
        if self.plateaued and not self._plateau_reported and self.best_epoch_estimate:
            self._plateau_reported = True
            logger.info(
                "[loss-watch] epoch %d: no image has been improving for %d epochs -- learning looks "
                "finished. Compare checkpoints around epoch %d.",
                epoch, self._no_improve_streak, self.best_epoch_estimate,
            )
        elif not self.plateaued:
            self._plateau_reported = False

        self._report(epoch, stats)
        return self.verdicts

    def _report(self, epoch: int, stats: dict[str, dict[str, Any]]) -> None:
        counts: dict[str, int] = {}
        for s in stats.values():
            counts[s["verdict"]] = counts.get(s["verdict"], 0) + 1
        order = ("easy", "mid", "learning", "watch", "suspect", "stuck", "exhausted")
        logger.info(
            "[loss-watch] epoch %d: %s%s", epoch,
            ", ".join(f"{v} {counts[v]}" for v in order if counts.get(v)),
            "" if self.apply_lr else " (per-image LR off: nothing is scaled)",
        )
        problems = sorted(
            (k for k, s in stats.items() if s["verdict"] in ("stuck", "suspect")),
            key=lambda k: -stats[k]["mean_residual"],
        )
        for k in problems[:8]:
            s = stats[k]
            logger.info(
                "[loss-watch]   %s: %s x%.2f (residual %+.4f)",
                os.path.basename(k), s["verdict"], s["multiplier"], s["mean_residual"],
            )
        if self.output_dir is None:
            return
        try:
            d = self.output_dir / "loss_log"
            d.mkdir(parents=True, exist_ok=True)
            payload = {
                "epoch": epoch,
                "per_image_lr": self.apply_lr,
                "plateaued": self.plateaued,
                "best_epoch_estimate": self.best_epoch_estimate,
                "counts": counts,
                "images": {
                    k: {f: (round(v, 6) if isinstance(v, float) else v) for f, v in s.items()}
                    for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["mean_residual"])
                },
            }
            tmp = d / "problem_images.json.tmp"
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, d / "problem_images.json")
        except OSError as e:
            logger.warning("[loss-watch] could not write problem_images.json: %s", e)

    # ---- resume --------------------------------------------------------

    _STATE_DICTS = ("_mult", "verdicts", "_stuck_votes", "_clear_votes", "_suspect_votes",
                    "_exhaust_votes", "_stuck_epochs", "_last_improving_epoch", "_improving_streak")

    def state_dict(self) -> dict:
        return {
            "rec": {k: {e: {b: list(c) for b, c in bm.items()} for e, bm in em.items()}
                    for k, em in self._rec.items()},
            "bsum": list(self._bsum), "bcnt": list(self._bcnt),
            "epochs_seen": sorted(self._epochs_seen),
            "confirmed_stuck": sorted(self._confirmed_stuck),
            "no_improve_streak": self._no_improve_streak,
            "plateau_reported": self._plateau_reported,
            "plateaued": self.plateaued,
            "best_epoch_estimate": self.best_epoch_estimate,
            **{name: dict(getattr(self, name)) for name in self._STATE_DICTS},
        }

    def load_state_dict(self, d: dict) -> None:
        if not d:
            return
        self._rec = {str(k): {int(e): {int(b): [float(c[0]), int(c[1])] for b, c in bm.items()}
                              for e, bm in em.items()}
                     for k, em in (d.get("rec") or {}).items()}
        self._bsum = [float(x) for x in d.get("bsum", self._bsum)]
        self._bcnt = [int(x) for x in d.get("bcnt", self._bcnt)]
        self._epochs_seen = {int(e) for e in d.get("epochs_seen", [])}
        self._confirmed_stuck = set(d.get("confirmed_stuck", []))
        self._no_improve_streak = int(d.get("no_improve_streak", 0))
        self._plateau_reported = bool(d.get("plateau_reported", False))
        self.plateaued = bool(d.get("plateaued", False))
        self.best_epoch_estimate = d.get("best_epoch_estimate")
        for name in self._STATE_DICTS:
            setattr(self, name, dict(d.get(name) or {}))


def build_loss_watch(args: Any, output_dir) -> PerImageLossWatch | None:
    """A watch when ``loss_watch`` or ``per_image_lr`` is on, otherwise None."""
    per_image_lr = bool(getattr(args, "per_image_lr", False))
    if not (per_image_lr or bool(getattr(args, "loss_watch", False))):
        return None
    logger.info("[loss-watch] per-image loss watch on (per_image_lr=%s)", per_image_lr)
    return PerImageLossWatch(output_dir, apply_lr=per_image_lr)
