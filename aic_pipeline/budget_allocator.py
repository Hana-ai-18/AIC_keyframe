"""
Tầng 3 — BUDGET ALLOCATION. Logic giữ nguyên (đã kiểm chứng qua test trước).
budget(shot) = base[class] + extra_per_second[class] * min(duration, cap_duration)
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

from .motion_scorer import MotionProfile
from .shot_detector import Shot


@dataclasses.dataclass
class ShotBudget:
    shot_id: int
    n_keyframes: int
    motion_class: str
    reason: str


DEFAULT_BASE = {"static": 1, "moderate": 2, "dynamic": 4}
DEFAULT_EXTRA_PER_SEC = {"static": 0.0, "moderate": 0.15, "dynamic": 0.4}
DEFAULT_CAP_DURATION = 12.0
DEFAULT_MAX_PER_SHOT = 10


def allocate_budget(
    shots: List[Shot],
    motion_profiles: List[MotionProfile],
    base: Optional[Dict[str, int]] = None,
    extra_per_second: Optional[Dict[str, float]] = None,
    cap_duration: float = DEFAULT_CAP_DURATION,
    max_per_shot: int = DEFAULT_MAX_PER_SHOT,
) -> List[ShotBudget]:
    base = base or DEFAULT_BASE
    extra_per_second = extra_per_second or DEFAULT_EXTRA_PER_SEC
    motion_by_id = {mp.shot_id: mp for mp in motion_profiles}
    budgets: List[ShotBudget] = []

    for shot in shots:
        mp = motion_by_id.get(shot.shot_id)
        cls = mp.motion_class if mp else "static"
        capped_duration = min(shot.duration, cap_duration)
        raw_budget = base[cls] + extra_per_second[cls] * capped_duration
        n = max(1, min(max_per_shot, round(raw_budget)))
        budgets.append(
            ShotBudget(
                shot_id=shot.shot_id, n_keyframes=n, motion_class=cls,
                reason=(f"class={cls}, duration={shot.duration:.1f}s "
                        f"(capped {capped_duration:.1f}s), base={base[cls]}, "
                        f"extra/s={extra_per_second[cls]}, raw={raw_budget:.2f} -> {n}"),
            )
        )
    return budgets


def allocate_budget_with_global_cap(
    shots: List[Shot],
    motion_profiles: List[MotionProfile],
    total_budget: int,
    base: Optional[Dict[str, int]] = None,
    extra_per_second: Optional[Dict[str, float]] = None,
    cap_duration: float = DEFAULT_CAP_DURATION,
) -> List[ShotBudget]:
    base = base or DEFAULT_BASE
    extra_per_second = extra_per_second or DEFAULT_EXTRA_PER_SEC
    motion_by_id = {mp.shot_id: mp for mp in motion_profiles}

    raw_list = []
    for shot in shots:
        mp = motion_by_id.get(shot.shot_id)
        cls = mp.motion_class if mp else "static"
        capped_duration = min(shot.duration, cap_duration)
        raw = base[cls] + extra_per_second[cls] * capped_duration
        raw_list.append((shot, cls, raw))

    total_raw = sum(r for _, _, r in raw_list)
    scale = 1.0 if total_raw <= total_budget else total_budget / total_raw

    budgets: List[ShotBudget] = []
    for shot, cls, raw in raw_list:
        n = max(1, round(raw * scale))
        budgets.append(
            ShotBudget(shot_id=shot.shot_id, n_keyframes=n, motion_class=cls,
                       reason=f"class={cls}, raw={raw:.2f}, scale={scale:.3f} -> {n}")
        )
    return budgets
