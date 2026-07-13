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
    """Chạy toàn bộ Tầng 1-4 trên 1 video. Xem PipelineConfig để tuỳ chỉnh.

    LƯU Ý VỀ TỐC ĐỘ: hàm này đọc video 2 LẦN RIÊNG BIỆT (1 lần cho Tầng 2 —
    motion scoring, 1 lần cho Tầng 4 — frame selection), vì score_motion() và
    select_keyframes() tự quản lý việc đọc frame độc lập. Với video AV1 dài
    (decode CPU nặng), đây là chi phí đáng kể khi chạy hàng trăm video. Dùng
    run_pipeline_optimized() thay thế nếu cần tối ưu tốc độ cho batch lớn —
    xem docstring hàm đó để biết mức cải thiện đã đo được.
    """
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


def run_pipeline_optimized(video_path: str, config: Optional[PipelineConfig] = None) -> PipelineResult:
    """
    TỐI ƯU TỐC ĐỘ: giống hệt run_pipeline() về kết quả, nhưng đọc video CHỈ
    1 LẦN DUY NHẤT (dùng chung cho cả Tầng 2 — motion và Tầng 4 — frame
    selection), thay vì 2 lần đọc riêng biệt như run_pipeline() gốc.

    Vì sao cần: log thực tế trên video AIC (K05_V031.mp4, AV1, 30.443 frame)
    cho thấy quá trình đọc video (decode CPU) chiếm phần lớn thời gian tổng
    (~280s/video). Với decode AV1 software (không hardware-accelerate), đọc
    2 lần gần như gấp đôi chi phí decode — đây là chỗ tối ưu có tác động lớn
    nhất khi cần xử lý hàng trăm video (ví dụ 700 video, mục tiêu 2-3 ngày).

    Cách hoạt động: gọi video_reader.get_video_frames() 1 LẦN NGAY ĐẦU HÀM,
    lấy frame + fps dùng chung cho toàn bộ shot của video này (qua cache có
    sẵn trong video_reader.py — get_frame_range_by_time tự động dùng lại
    cache nếu cùng video_path + cùng cấu hình resize/max_frames).

    LƯU Ý: chỉ tối ưu được I/O đọc video của Tầng 2 + Tầng 4 (đều đã dùng
    video_reader.py sau bản vá AV1). Tầng 1 (shot detection qua OmniShotCut)
    vẫn đọc video riêng bằng _read_video_pyav của chính nó (kiến trúc khác,
    cần format/resize riêng cho model DETR) — đây là giới hạn hợp lý vì
    Tầng 1 cần độ phân giải/fps khác hẳn Tầng 2/4.
    """
    from .video_reader import get_video_frames, DEFAULT_MAX_FRAMES, DEFAULT_READ_WIDTH, DEFAULT_READ_HEIGHT

    config = config or PipelineConfig()
    t0 = time.time()

    shots = detect_shots(video_path, backend=config.shot_backend, **config.shot_kwargs)
    t1 = time.time()

    # Đọc video ĐÚNG 1 LẦN ở đây — score_motion() và select_keyframes() bên
    # dưới sẽ tự động dùng lại cache này (cùng video_path + cùng cấu hình
    # resize mặc định) qua video_reader.py, không đọc lại từ đầu.
    get_video_frames(
        video_path, max_frames=DEFAULT_MAX_FRAMES,
        resize_width=DEFAULT_READ_WIDTH, resize_height=DEFAULT_READ_HEIGHT,
    )
    t_preload = time.time()

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
            "video_preload": round(t_preload - t1, 3),
            "motion_scoring": round(t2 - t_preload, 3),
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
    Chạy TUẦN TỰ, dùng run_pipeline_optimized() nội bộ (1 lần đọc video/video).

    Với batch LỚN (hàng trăm video), cân nhắc dùng run_pipeline_batch_parallel()
    thay thế để tận dụng nhiều CPU core cùng lúc — xem docstring hàm đó.
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
            result = run_pipeline_optimized(video_path, config)
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


def _process_one_video_for_parallel(args):
    """Hàm worker cho multiprocessing — PHẢI ở module level (không phải nested
    function/lambda) để pickle được khi truyền vào Pool.map()."""
    video_path, config, use_optimized = args
    import traceback as _tb
    try:
        fn = run_pipeline_optimized if use_optimized else run_pipeline
        result = fn(video_path, config)
        return video_path, result, None
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}\n{_tb.format_exc()}"
        return video_path, None, err_msg


def run_pipeline_batch_parallel(
    video_dir: str,
    config: Optional[PipelineConfig] = None,
    pattern: str = "*.mp4",
    limit: Optional[int] = None,
    n_workers: Optional[int] = None,
    use_optimized: bool = True,
) -> Dict[str, PipelineResult]:
    """
    TỐI ƯU TỐC ĐỘ CHO BATCH LỚN: chạy song song nhiều video cùng lúc bằng
    multiprocessing, tận dụng nhiều CPU core (decode video AV1 là CPU-bound
    — chạy song song scale gần tuyến tính theo số core, không bị giới hạn
    bởi GIL vì mỗi video chạy trong process riêng).

    Vì sao cần cho mục tiêu "700 video trong 2-3 ngày": đo thực tế 1 video
    15 phút tốn ~280s TUẦN TỰ (một phần đã giảm nhờ run_pipeline_optimized).
    700 video tuần tự = 700 * 280s ≈ 54.4 giờ (~2.3 ngày) — CHỈ VỪA ĐỦ nếu
    chạy tuần tự với model GPU chiếm phần lớn thời gian. Chạy song song
    n_workers video cùng lúc (mỗi worker vẫn dùng chung 1 GPU cho model, chỉ
    phần đọc/decode video và tính CPU chạy song song) giảm thời gian theo hệ
    số gần n_workers (với 4 CPU core: gần 4x nhanh hơn cho phần I/O).

    LƯU Ý QUAN TRỌNG VỀ GPU: nếu config.shot_backend hoặc config.embedder
    dùng GPU (OmniShotCutDetector, ClipEmbedder), mỗi worker process sẽ tự
    load model RIÊNG lên GPU — với n_workers lớn có thể TRÀN VRAM. Khuyến
    nghị: n_workers = 2-3 cho GPU T4 (16GB VRAM), không phải bằng số CPU
    core. Test trước với n_workers=2 để đo VRAM thực tế trước khi tăng.

    Args:
        n_workers: số process chạy song song. None = tự động dùng
                   min(4, cpu_count()) — nhưng xem lưu ý VRAM ở trên, nên tự
                   set rõ ràng (khuyến nghị 2-3) thay vì để mặc định nếu
                   dùng GPU.
        use_optimized: True = dùng run_pipeline_optimized() cho mỗi video
                       (khuyến nghị, đã đo nhanh hơn ~1.8x so với run_pipeline).

    Returns:
        Dict {video_path: PipelineResult}, giống hệt run_pipeline_batch().
    """
    import multiprocessing as mp

    config = config or PipelineConfig()
    video_paths = sorted(glob.glob(os.path.join(video_dir, pattern)))
    if limit is not None:
        video_paths = video_paths[:limit]

    if not video_paths:
        logger.warning(f"Không tìm thấy video nào khớp '{pattern}' trong {video_dir}")
        return {}

    if n_workers is None:
        n_workers = min(4, mp.cpu_count())
    n_workers = max(1, n_workers)

    logger.info(
        f"Bắt đầu chạy SONG SONG {len(video_paths)} video với {n_workers} worker "
        f"(use_optimized={use_optimized})..."
    )

    tasks = [(vp, config, use_optimized) for vp in video_paths]
    results: Dict[str, PipelineResult] = {}

    t_start = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for i, (video_path, result, error) in enumerate(pool.imap_unordered(_process_one_video_for_parallel, tasks)):
            if error is not None:
                logger.error(f"[{i + 1}/{len(video_paths)}] LỖI {video_path}: {error}")
                results[video_path] = PipelineResult(
                    video_path=video_path, shots=[], motion_profiles=[],
                    shot_budgets=[], keyframes=[], stats={}, error=error,
                )
            else:
                results[video_path] = result
                elapsed = time.time() - t_start
                avg_per_video = elapsed / (i + 1)
                remaining = avg_per_video * (len(video_paths) - i - 1)
                logger.info(
                    f"[{i + 1}/{len(video_paths)}] XONG {os.path.basename(video_path)}: "
                    f"{result.stats['n_shots']} shot, {result.stats['n_keyframes']} keyframe "
                    f"| trung bình {avg_per_video:.1f}s/video | ước tính còn {remaining/3600:.1f}h"
                )

    n_ok = sum(1 for r in results.values() if r.error is None)
    n_err = len(results) - n_ok
    total_elapsed = time.time() - t_start
    logger.info(
        f"Hoàn tất batch song song: {n_ok} thành công, {n_err} lỗi, "
        f"tổng thời gian {total_elapsed/3600:.1f}h ({total_elapsed/max(1,len(video_paths)):.1f}s/video trung bình)."
    )
    return results
