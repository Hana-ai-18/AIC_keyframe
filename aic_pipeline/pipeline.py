"""
PIPELINE ĐIỀU PHỐI (offline, Tầng 1-4) — nâng cấp so với bản trước:
  - Hỗ trợ feature_mode="semantic" (dùng CLIP embedding thật trên GPU) cho Tầng 4.
  - Thêm run_pipeline_batch() để chạy hàng loạt trên cả THƯ MỤC video (data crawl),
    kèm xử lý lỗi từng file (1 video lỗi không làm sập cả batch) và log tiến độ.

Tầng 5 (temporal_reranker) KHÔNG nằm trong luồng offline này — nó chạy ở
QUERY-TIME, trên kết quả retrieval đã có (xem temporal_reranker.py).
"""
from __future__ import annotations

import dataclasses
import glob
import logging
import os
import time
import traceback
from typing import Dict, List, Optional

from .budget_allocator import ShotBudget, allocate_budget, allocate_budget_with_global_cap
from .frame_selector import Keyframe, select_keyframes
from .motion_scorer import MotionProfile, score_motion
from .shot_detector import Shot, ShotDetectorBackend, detect_shots

logger = logging.getLogger("aic_pipeline.pipeline")


@dataclasses.dataclass
class PipelineConfig:
    # Tầng 1
    shot_backend: Optional[ShotDetectorBackend] = None
    shot_kwargs: Dict = dataclasses.field(default_factory=dict)

    # Tầng 2
    static_threshold: float = 0.5
    dynamic_threshold: float = 2.0
    motion_resize_to: tuple = (160, 90)
    motion_frame_stride: int = 2

    # Tầng 3
    budget_base: Optional[Dict[str, int]] = None
    budget_extra_per_second: Optional[Dict[str, float]] = None
    budget_cap_duration: float = 12.0
    budget_max_per_shot: int = 10
    global_budget: Optional[int] = None

    # Tầng 4
    feature_mode: str = "cheap"     # "cheap" | "semantic"
    embedder: Optional[object] = None   # instance ClipEmbedder, bắt buộc nếu feature_mode="semantic"
    frame_quality_weight: float = 0.3
    frame_min_sharpness_percentile: float = 15.0
    frame_stride: int = 1
    store_images: bool = True
    store_embeddings: bool = False


@dataclasses.dataclass
class PipelineResult:
    video_path: str
    shots: List[Shot]
    motion_profiles: List[MotionProfile]
    shot_budgets: List[ShotBudget]
    keyframes: List[Keyframe]
    stats: Dict = dataclasses.field(default_factory=dict)
    error: Optional[str] = None   # chỉ set khi dùng run_pipeline_batch và video này lỗi


def run_pipeline(video_path: str, config: Optional[PipelineConfig] = None) -> PipelineResult:
    """Chạy toàn bộ Tầng 1-4 trên 1 video. Xem PipelineConfig để tuỳ chỉnh."""
    config = config or PipelineConfig()
    t0 = time.time()

    shots = detect_shots(video_path, backend=config.shot_backend, **config.shot_kwargs)
    t1 = time.time()

    motion_profiles = score_motion(
        video_path, shots,
        static_threshold=config.static_threshold,
        dynamic_threshold=config.dynamic_threshold,
        resize_to=config.motion_resize_to,
        frame_stride=config.motion_frame_stride,
    )
    t2 = time.time()

    if config.global_budget is not None:
        shot_budgets = allocate_budget_with_global_cap(
            shots, motion_profiles, total_budget=config.global_budget,
            base=config.budget_base, extra_per_second=config.budget_extra_per_second,
            cap_duration=config.budget_cap_duration,
        )
    else:
        shot_budgets = allocate_budget(
            shots, motion_profiles,
            base=config.budget_base, extra_per_second=config.budget_extra_per_second,
            cap_duration=config.budget_cap_duration, max_per_shot=config.budget_max_per_shot,
        )
    t3 = time.time()

    budget_by_shot = {b.shot_id: b.n_keyframes for b in shot_budgets}
    keyframes: List[Keyframe] = []
    for shot in shots:
        n = budget_by_shot.get(shot.shot_id, 1)
        kfs = select_keyframes(
            video_path, shot, n_keyframes=n,
            feature_mode=config.feature_mode,
            embedder=config.embedder,
            quality_weight=config.frame_quality_weight,
            min_sharpness_percentile=config.frame_min_sharpness_percentile,
            frame_stride=config.frame_stride,
            store_images=config.store_images,
            store_embeddings=config.store_embeddings,
        )
        keyframes.extend(kfs)
    t4 = time.time()

    motion_class_counts: Dict[str, int] = {}
    for mp in motion_profiles:
        motion_class_counts[mp.motion_class] = motion_class_counts.get(mp.motion_class, 0) + 1

    stats = {
        "n_shots": len(shots),
        "n_keyframes": len(keyframes),
        "motion_class_distribution": motion_class_counts,
        "avg_keyframes_per_shot": len(keyframes) / max(1, len(shots)),
        "feature_mode": config.feature_mode,
        "timing_seconds": {
            "shot_detection": round(t1 - t0, 3),
            "motion_scoring": round(t2 - t1, 3),
            "budget_allocation": round(t3 - t2, 3),
            "frame_selection": round(t4 - t3, 3),
            "total": round(t4 - t0, 3),
        },
    }

    return PipelineResult(
        video_path=video_path, shots=shots, motion_profiles=motion_profiles,
        shot_budgets=shot_budgets, keyframes=keyframes, stats=stats,
    )


def run_pipeline_batch(
    video_dir: str,
    config: Optional[PipelineConfig] = None,
    pattern: str = "*.mp4",
    limit: Optional[int] = None,
    continue_on_error: bool = True,
) -> Dict[str, PipelineResult]:
    """
    Chạy pipeline trên TẤT CẢ video trong 1 thư mục — dùng cho data tự crawl.

    Args:
        video_dir: thư mục chứa video (ví dụ "/kaggle/working/data_crawl").
        pattern: glob pattern để lọc file, mặc định "*.mp4".
        limit: giới hạn số video xử lý (hữu ích để test nhanh trên vài video
               trước khi chạy hết thư mục lớn).
        continue_on_error: True thì 1 video lỗi (file hỏng, codec lạ...) không
                           làm dừng cả batch — lỗi được ghi vào
                           PipelineResult.error, bạn xem lại sau.

    Returns:
        Dict {video_path: PipelineResult}. Nếu video lỗi, PipelineResult sẽ có
        error != None và các trường khác rỗng/mặc định.
    """
    config = config or PipelineConfig()
    video_paths = sorted(glob.glob(os.path.join(video_dir, pattern)))
    if limit is not None:
        video_paths = video_paths[:limit]

    if not video_paths:
        logger.warning(f"Không tìm thấy video nào khớp '{pattern}' trong {video_dir}")
        return {}

    logger.info(f"Bắt đầu chạy batch trên {len(video_paths)} video...")
    results: Dict[str, PipelineResult] = {}

    for i, video_path in enumerate(video_paths):
        logger.info(f"[{i + 1}/{len(video_paths)}] Xử lý: {video_path}")
        try:
            result = run_pipeline(video_path, config)
            results[video_path] = result
            logger.info(
                f"  -> {result.stats['n_shots']} shot, "
                f"{result.stats['n_keyframes']} keyframe, "
                f"{result.stats['timing_seconds']['total']:.1f}s"
            )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"  LỖI khi xử lý {video_path}: {err_msg}")
            if not continue_on_error:
                raise
            results[video_path] = PipelineResult(
                video_path=video_path, shots=[], motion_profiles=[],
                shot_budgets=[], keyframes=[], stats={}, error=err_msg,
            )

    n_ok = sum(1 for r in results.values() if r.error is None)
    n_err = len(results) - n_ok
    logger.info(f"Hoàn tất batch: {n_ok} thành công, {n_err} lỗi.")
    return results
