"""Progressive pruning with optional regrowth.

The pruner owns *which* connections exist; the engine and optimizer own what
that means (exact zeros forward, zero gradient, state reset on a flip). All
the pruner ever does to the network is call `Parameter.set_mask`.

Per training step the loop calls:

    if pruner.is_update_step(step): pruner.update(step)   # before the forward pass
    ... forward, backward ...
    pruner.after_backward()                               # accumulate gradient statistics
    optimizer.step()

Masks change *before* a forward pass, so the gradients of a step always
belong to the mask that step was computed with.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from engine import Parameter

from .criteria import CRITERIA, GradStats, dense_gradient, regrowth_score
from .schedule import SCHEDULES


def select(scores: Sequence[np.ndarray], candidates: Sequence[np.ndarray], k: int, largest: bool) -> list:
    """Pick exactly `k` coordinates with the lowest (or largest) score among `candidates`.

    `scores` and `candidates` (boolean) are parallel lists of arrays, ranked
    jointly as one pool. Returns one boolean array per input marking the
    chosen coordinates.
    """
    chosen = [np.zeros(c.shape, dtype=bool) for c in candidates]
    if k <= 0:
        return chosen
    pool = np.concatenate([s[c] for s, c in zip(scores, candidates)])
    if k > pool.size:
        raise ValueError(f"asked for {k} coordinates but only {pool.size} are candidates")
    if largest:
        pool = -pool
    picked = np.zeros(pool.size, dtype=bool)
    picked[np.argpartition(pool, k - 1)[:k]] = True  # exactly k, ties broken deterministically
    offset = 0
    for out, cand in zip(chosen, candidates):
        n = int(cand.sum())
        out[cand] = picked[offset : offset + n]  # boolean indexing visits coordinates in the same order as s[c]
        offset += n
    return chosen


class Pruner:
    """Drive a set of weight tensors to `target_sparsity` along a schedule.

    scope="global": all tensors compete in one ranking (scores must be comparable across layers).
    scope="layer":  every tensor is pruned to the same sparsity on its own ranking.

    regrow_fraction > 0 enables regrowth: at each update, after pruning to the
    scheduled budget, a further fraction of the active connections is dropped
    and the same number of masked connections is revived (highest
    `regrowth_score`), so the budget is unchanged. The fraction follows a
    cosine decay to zero at `end_step`: exploration early, a settled mask late.
    """

    def __init__(
        self,
        params: Sequence[Parameter],
        criterion: str = "fisher",
        target_sparsity: float = 0.9,
        schedule: str = "cubic",
        start_step: int = 0,
        end_step: int = 1000,
        interval: int = 100,
        scope: str = "global",
        regrow_fraction: float = 0.0,
        batch_size: int = 128,
        stats_decay: Optional[float] = 0.99,
        seed: int = 0,
    ):
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        if scope not in ("global", "layer"):
            raise ValueError(f"unknown scope {scope!r}")
        if end_step < start_step or interval < 1:
            raise ValueError("need start_step <= end_step and interval >= 1")
        self.params = list(params)
        self.criterion_name = criterion
        self.criterion = CRITERIA[criterion]
        self.target_sparsity = target_sparsity
        self.schedule = SCHEDULES[schedule]
        self.start_step, self.end_step, self.interval = start_step, end_step, interval
        self.scope = scope
        self.regrow_fraction = regrow_fraction
        self.rng = np.random.default_rng(seed)
        self.total = sum(p.size for p in self.params)

        self.needs_stats = self.criterion.needs_gradients or regrow_fraction > 0
        self.stats = [GradStats(p.shape, batch_size, stats_decay) for p in self.params] if self.needs_stats else None
        self.events: list = []
        # Regrowth bookkeeping for the stability analysis.
        self.times_revived = [np.zeros(p.shape, dtype=np.int32) for p in self.params]
        self.times_pruned = [np.zeros(p.shape, dtype=np.int32) for p in self.params]

    # -------------------------------------------------------------- queries

    @property
    def num_active(self) -> int:
        return sum(p.num_active for p in self.params)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.num_active / self.total

    def layer_sparsity(self) -> list:
        return [p.sparsity for p in self.params]

    def is_update_step(self, step: int) -> bool:
        if not self.start_step <= step <= self.end_step:
            return False
        return (step - self.start_step) % self.interval == 0 or step == self.end_step

    @staticmethod
    def budget(size: int, sparsity: float) -> int:
        """Connections allowed to remain: the largest integer <= (1 - sparsity) * size."""
        return int(math.floor((1.0 - sparsity) * size + 1e-9))

    # ------------------------------------------------------------- training

    def after_backward(self) -> None:
        if self.needs_stats:
            for p, stats in zip(self.params, self.stats):
                stats.update(dense_gradient(p), p.data)

    def scores(self) -> list:
        """Importance of every coordinate (only active ones are ever compared)."""
        if self.criterion_name == "random":
            return [self.rng.random(p.shape) for p in self.params]
        stats = self.stats if self.criterion.needs_gradients else [None] * len(self.params)
        return [self.criterion.score(p.data, s) for p, s in zip(self.params, stats)]

    def update(self, step: int) -> dict:
        """Move the masks to this step's scheduled sparsity. Returns a record of what changed."""
        scheduled = self.schedule(step, self.start_step, self.end_step, self.target_sparsity)
        progress = (step - self.start_step) / max(1, self.end_step - self.start_step)
        regrow_now = self.regrow_fraction * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scores = self.scores()
        old = [np.ones(p.shape, dtype=bool) if p.mask is None else p.mask for p in self.params]
        new = [m.copy() for m in old]
        groups = [list(range(len(self.params)))] if self.scope == "global" else [[i] for i in range(len(self.params))]
        removed_score = 0.0
        n_regrown = 0

        for group in groups:
            size = sum(self.params[i].size for i in group)
            active = sum(int(new[i].sum()) for i in group)

            # 1. Prune the least important active connections down to the budget.
            n_drop = active - self.budget(size, scheduled)
            drop = select([scores[i] for i in group], [new[i] for i in group], n_drop, largest=False)
            for i, d in zip(group, drop):
                new[i] &= ~d
                removed_score += float(scores[i][d].sum())

            # 2. Regrowth: swap k more weak active connections for the k most promising masked ones.
            #    Candidates are connections that were already masked before this update.
            if regrow_now > 0:
                n_candidates = sum(int((~old[i]).sum()) for i in group)
                k = min(int(regrow_now * (active - max(0, n_drop))), n_candidates)
                extra = select([scores[i] for i in group], [new[i] for i in group], k, largest=False)
                grow = select(
                    [regrowth_score(self.stats[i]) for i in group], [~old[i] for i in group], k, largest=True
                )
                for i, e, g in zip(group, extra, grow):
                    new[i] &= ~e
                    new[i] |= g
                    removed_score += float(scores[i][e].sum())
                n_regrown += k

        n_pruned = 0
        for p, before, after, revived, pruned in zip(self.params, old, new, self.times_revived, self.times_pruned):
            newly_pruned, newly_revived = before & ~after, ~before & after
            n_pruned += int(newly_pruned.sum())
            pruned += newly_pruned
            revived += newly_revived
            if newly_pruned.any() or newly_revived.any():
                p.set_mask(after)  # zeroes pruned weights; revived ones restart from 0 with fresh optimizer state

        event = {
            "step": step,
            "scheduled_sparsity": scheduled,
            "sparsity": self.sparsity,
            "layer_sparsity": self.layer_sparsity(),
            "pruned": n_pruned,
            "regrown": n_regrown,
            "removed_score": removed_score,
        }
        self.events.append(event)
        return event

    # ------------------------------------------------------------- analysis

    def regrowth_summary(self) -> dict:
        """What became of regrown connections (all zeros when regrowth is off)."""
        final = [np.ones(p.shape, dtype=bool) if p.mask is None else p.mask for p in self.params]
        ever_revived = [r > 0 for r in self.times_revived]
        return {
            "regrow_events": int(sum(int(r.sum()) for r in self.times_revived)),
            "connections_ever_regrown": int(sum(int(e.sum()) for e in ever_revived)),
            "regrown_and_alive_at_end": int(sum(int((e & f).sum()) for e, f in zip(ever_revived, final))),
            "pruned_more_than_once": int(sum(int((t > 1).sum()) for t in self.times_pruned)),
            "final_active": int(sum(int(f.sum()) for f in final)),
        }
