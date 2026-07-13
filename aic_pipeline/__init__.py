"""
aic_pipeline — Pipeline cắt & lọc keyframe + temporal reranking cho video retrieval
(HCM AI Challenge / VBS). Thiết kế chạy trên Kaggle (GPU A100) với data tự crawl.

5 TẦNG:
  1. shot_detector      — cắt shot boundary (AutoShot-ready hook + baseline cổ điển)
  2. motion_scorer       — đo mật độ chuyển động từng shot (optical flow, CPU)
  3. budget_allocator    — phân bổ số keyframe/shot theo motion (tĩnh ít, động nhiều)
  4. frame_selector      — chọn frame cụ thể: lọc chất lượng + CLIP/BEiT-3 embedding
                            (vá lỗ hổng: dùng embedding ngữ nghĩa thay vì chỉ màu/cạnh)
                            + farthest-point-sampling đa dạng
  5. temporal_reranker   — nối chuỗi truy vấn nhiều giai đoạn, relinking toàn cục
                            (lấy cơ chế của WESp, generalize thành module độc lập)

query_aware.py — chấm điểm frame theo câu hỏi thật bằng MLLM, CHỈ gọi lúc có
                  query (không caption offline tràn lan).

embeddings.py  — wrapper load CLIP/BEiT-3/SigLIP2 qua HuggingFace, dùng chung
                  cho Tầng 4 (lọc semantic) và có thể tái dùng cho index chính thức.
"""

from .shot_detector import (
    detect_shots, Shot, GPUShotDetector, HistogramSSIMDetector, AutoShotDetector,
    OmniShotCutDetector, make_omnishotcut_detector,
)
from .motion_scorer import score_motion, MotionProfile, calibrate_thresholds, auto_calibrate_thresholds_gmm
from .budget_allocator import allocate_budget, allocate_budget_with_global_cap, ShotBudget
from .frame_selector import select_keyframes, Keyframe
from .video_reader import get_video_frames, get_frame_range_by_time, clear_cache as clear_video_cache
from .temporal_reranker import TemporalReranker, StageHit, ChainResult
from .object_graph import (
    build_object_graph, summarize_graph_for_query, ObjectGraph, ObjectTrack,
    ObjectGraphEdge, DetectedObject, ContourFallbackDetector, YoloDetector,
)
from .pipeline import run_pipeline, run_pipeline_batch, PipelineConfig, PipelineResult
from .streaming_batch import run_pipeline_fast, FastPipelineConfig, FastPipelineResult

__all__ = [
    "detect_shots", "Shot", "GPUShotDetector", "HistogramSSIMDetector", "AutoShotDetector",
    "OmniShotCutDetector", "make_omnishotcut_detector",
    "score_motion", "MotionProfile", "calibrate_thresholds", "auto_calibrate_thresholds_gmm",
    "allocate_budget", "allocate_budget_with_global_cap", "ShotBudget",
    "select_keyframes", "Keyframe",
    "get_video_frames", "get_frame_range_by_time", "clear_video_cache",
    "TemporalReranker", "StageHit", "ChainResult",
    "build_object_graph", "summarize_graph_for_query", "ObjectGraph", "ObjectTrack",
    "ObjectGraphEdge", "DetectedObject", "ContourFallbackDetector", "YoloDetector",
    "run_pipeline", "run_pipeline_batch", "PipelineConfig", "PipelineResult",
    "run_pipeline_fast", "FastPipelineConfig", "FastPipelineResult",
]

__version__ = "0.6.0"
