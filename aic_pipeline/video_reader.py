"""
VIDEO READER — đọc frame video dùng chung cho toàn pipeline (Tầng 2, 3, 4).

VÁ BUG NGHIÊM TRỌNG: các tầng 2 (motion_scorer), 4 (frame_selector) và
streaming_batch.py đều dùng cv2.VideoCapture (OpenCV) để đọc video — nhưng
OpenCV KHÔNG hỗ trợ codec AV1 (dataset AIC dùng AV1). Khi Tầng 1
(OmniShotCutDetector, đã patch dùng PyAV) cắt shot thành công trên video
AV1, Tầng 4 dùng OpenCV để trích frame trong mỗi shot lại LUÔN THẤT BẠI ÂM
THẦM (cv2.VideoCapture.read() trả về ret=False ngay từ frame đầu, vòng lặp
break tức thì) — kết quả: n_shots > 0 nhưng n_keyframes = 0.

Giải pháp: đọc TOÀN BỘ video một lần bằng PyAV (hỗ trợ AV1 qua fallback
software decode), lưu vào cache theo video_path. Các tầng khác lấy frame
theo index từ cache này thay vì tự mở lại video bằng OpenCV — vừa sửa đúng
bug AV1, vừa NHANH HƠN nhiều (đọc video chỉ 1 lần cho toàn pipeline).
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger("aic_pipeline.video_reader")

_cache: Dict[str, Tuple[np.ndarray, float]] = {}
_CACHE_MAX_VIDEOS = 1  # chỉ giữ 1 video gần nhất trong RAM


def _read_all_frames_pyav(video_path: str, max_frames: int = 20000) -> Tuple[np.ndarray, float]:
    """Đọc toàn bộ frame bằng PyAV (hỗ trợ AV1), giữ độ phân giải gốc, trả BGR."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) or 25.0
    stream.thread_type = "AUTO"

    frames = []
    consecutive_empty = 0
    max_consecutive_empty = 200

    for packet in container.demux(video=0):
        try:
            decoded = packet.decode()
        except Exception as e:
            logger.debug(f"Bỏ qua packet lỗi decode: {e}")
            decoded = []

        if not decoded:
            consecutive_empty += 1
            if consecutive_empty >= max_consecutive_empty:
                logger.warning(
                    f"Phát hiện {consecutive_empty} packet liên tiếp lỗi/rỗng "
                    f"(stream có thể hỏng một phần) — dừng sớm với {len(frames)} frame."
                )
                break
            continue
        consecutive_empty = 0

        for frame in decoded:
            try:
                img_rgb = frame.to_ndarray(format="rgb24")
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.debug(f"Bỏ qua frame lỗi convert: {e}")
                continue
            frames.append(img_bgr)
            if len(frames) >= max_frames:
                logger.warning(f"Đã đạt max_frames={max_frames} khi đọc full-res, dừng sớm.")
                break
        if len(frames) >= max_frames:
            break

    container.close()

    if not frames:
        raise ValueError(f"Không đọc được frame nào từ video: {video_path}")
    return np.stack(frames, axis=0), fps


def get_video_frames(video_path: str, max_frames: int = 20000) -> Tuple[np.ndarray, float]:
    """
    Entry point chính — trả về (frames_bgr (T,H,W,3) uint8, fps), có cache.
    Dùng cho Tầng 2, Tầng 4 — thay cv2.VideoCapture riêng lẻ. Cache đảm bảo
    video chỉ được decode ĐÚNG 1 LẦN dù gọi từ nhiều shot khác nhau.
    """
    if video_path in _cache:
        return _cache[video_path]

    frames, fps = _read_all_frames_pyav(video_path, max_frames=max_frames)

    if len(_cache) >= _CACHE_MAX_VIDEOS:
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]
        logger.debug(f"Giải phóng cache video cũ: {oldest_key}")

    _cache[video_path] = (frames, fps)
    return frames, fps


def get_frame_range(video_path: str, start_frame: int, end_frame: int, max_frames: int = 20000):
    """Trả về (indices, frames_bgr) cho khoảng [start_frame, end_frame) —
    CHỈ DÙNG khi start_frame/end_frame chắc chắn cùng hệ quy chiếu fps với
    video gốc (ví dụ HistogramSSIMDetector/AutoShotDetector). KHÔNG dùng với
    OmniShotCutDetector — xem get_frame_range_by_time() bên dưới."""
    frames, fps = get_video_frames(video_path, max_frames=max_frames)
    total = frames.shape[0]
    s = max(0, min(start_frame, total))
    e = max(0, min(end_frame, total))
    if s >= e:
        return [], np.zeros((0, *frames.shape[1:]), dtype=frames.dtype)
    indices = list(range(s, e))
    return indices, frames[s:e]


def get_frame_range_by_time(video_path: str, start_time: float, end_time: float, max_frames: int = 20000):
    """
    Trả về (indices, frames_bgr) cho khoảng THỜI GIAN [start_time, end_time)
    giây — dùng THAY get_frame_range() khi Shot.start_frame/end_frame có thể
    KHÔNG cùng hệ quy chiếu fps với video gốc.

    QUAN TRỌNG: OmniShotCutDetector tính Shot.start_frame/end_frame theo
    "fps hiệu dụng" đã bị subsample (ví dụ 6fps sau khi giảm từ 30fps gốc để
    tránh tràn RAM — xem _read_video_pyav trong _omnishotcut_vendor). Nếu
    dùng thẳng start_frame/end_frame để index vào mảng frame đọc ở FPS GỐC
    (không subsample) của video_reader này, vị trí sẽ SAI HOÀN TOÀN (lệch
    hệ quy chiếu). Nhưng Shot.start_time/end_time (giây) LUÔN ĐÚNG vì được
    tính bằng frame_idx / fps_hiệu_dụng — tự triệt tiêu đúng thời gian thực.
    Do đó hàm này quy đổi lại theo THỜI GIAN THỰC (giây) sang fps GỐC của
    video_reader, đảm bảo đúng vị trí bất kể detector nào tạo ra Shot.
    """
    frames, fps = get_video_frames(video_path, max_frames=max_frames)
    total = frames.shape[0]
    start_frame_real = int(round(start_time * fps))
    end_frame_real = int(round(end_time * fps))
    s = max(0, min(start_frame_real, total))
    e = max(0, min(end_frame_real, total))
    if s >= e:
        e = min(s + 1, total)  # đảm bảo lấy ít nhất 1 frame nếu shot cực ngắn
    if s >= e:
        return [], np.zeros((0, *frames.shape[1:]), dtype=frames.dtype)
    indices = list(range(s, e))
    return indices, frames[s:e]


def clear_cache():
    """Xoá toàn bộ cache thủ công — gọi khi cần giải phóng RAM ngay lập tức."""
    _cache.clear()
