"""
VIDEO READER — đọc frame video dùng chung cho toàn pipeline (Tầng 2, 3, 4).

VÁ BUG NGHIÊM TRỌNG #1 (AV1): các tầng 2 (motion_scorer), 4 (frame_selector)
đều dùng cv2.VideoCapture (OpenCV) để đọc video — nhưng OpenCV KHÔNG hỗ trợ
codec AV1 (dataset AIC dùng AV1), khiến cap.read() luôn trả ret=False, mọi
shot bị gán nhầm "static" và select_keyframes() luôn rỗng (n_keyframes=0).
Đã vá bằng cách đọc qua PyAV (hỗ trợ AV1).

VÁ BUG NGHIÊM TRỌNG #2 (TRÀN RAM — phát hiện khi test trên video AIC thật,
K05_V030.mp4, 1920x1080, 32.714 frame): bản vá #1 ban đầu đọc GIỮ NGUYÊN ĐỘ
PHÂN GIẢI GỐC — với max_frames=6000 ở 1920x1080x3 bytes = 37.3GB RAM, vượt
xa RAM Kaggle (13-16GB), khiến process bị hệ điều hành kill ("Killed", không
có traceback Python vì bị kill ở tầng OS, không phải exception thường).
Đã vá bằng CHIẾN LƯỢC GIẢM CHẤT LƯỢNG BẮT BUỘC (không phải tuỳ chọn):
  1. RESIZE xuống độ phân giải nhỏ khi đọc (mặc định 480x270 — đủ để tính
     sharpness/exposure/optical-flow/CLIP chính xác, không cần full-res).
  2. Giảm max_frames mặc định xuống mức thực sự an toàn (2000, không phải
     6000/20000 như trước — những con số đó CHƯA tính đúng RAM thật).
  3. Hàm estimate_ram_gb() để tự kiểm tra trước khi đọc, cảnh báo rõ ràng
     nếu cấu hình vẫn có nguy cơ tràn RAM thay vì để process bị kill âm thầm.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("aic_pipeline.video_reader")

_cache: Dict[str, Tuple[np.ndarray, float]] = {}
_CACHE_MAX_VIDEOS = 1  # chỉ giữ 1 video gần nhất trong RAM

# Độ phân giải mặc định khi đọc frame cho Tầng 2/4 — ĐỦ để tính sharpness,
# exposure, optical flow, và feed vào CLIP (CLIP tự resize về 224x224 nội
# bộ nên không cần giữ full-res tới lúc đó). 480x270 giữ đủ chi tiết để
# đánh giá "frame nào nét/mờ" chính xác, không cần 1920x1080 gốc.
DEFAULT_READ_WIDTH = 480
DEFAULT_READ_HEIGHT = 270
DEFAULT_MAX_FRAMES = 2000


def estimate_ram_gb(n_frames: int, width: int, height: int) -> float:
    """Tính RAM (GB) cần thiết để giữ n_frames ảnh BGR uint8 kích thước width x height."""
    return n_frames * height * width * 3 / 1e9


def _read_all_frames_pyav(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resize_width: Optional[int] = DEFAULT_READ_WIDTH,
    resize_height: Optional[int] = DEFAULT_READ_HEIGHT,
) -> Tuple[np.ndarray, float]:
    """
    Đọc frame bằng PyAV (hỗ trợ AV1), RESIZE NGAY TRONG LÚC ĐỌC.

    ĐÃ VÁ BUG NGHIÊM TRỌNG #2 (phát hiện khi hỏi về đánh đổi max_frames):
    bản trước dừng TUẦN TỰ ngay khi đủ max_frames — với video dài (18 phút,
    30fps), max_frames=2000 chỉ đọc được ~67 GIÂY ĐẦU rồi bỏ hẳn phần còn
    lại (17 phút cuối video KHÔNG được xử lý, không có shot/keyframe nào ở
    đó). Đây là lỗi nghiêm trọng hơn cả việc giảm độ phân giải.

    Đã sửa bằng SUBSAMPLE CÁCH QUÃNG ĐỀU (giống cơ chế OmniShotCut dùng cho
    Tầng 1): ước tính tổng số frame trước (qua stream.frames hoặc duration*fps),
    tính frame_stride = total // max_frames, CHỈ GIỮ 1 frame mỗi frame_stride
    — đảm bảo max_frames frame được chọn TRẢI ĐỀU khắp toàn bộ video, không
    riêng đoạn đầu. fps trả về được điều chỉnh tương ứng (fps / frame_stride)
    để Shot.start_time/end_time (tính qua get_frame_range_by_time) vẫn đúng
    thời gian thực.
    """
    import av

    est_gb = estimate_ram_gb(max_frames, resize_width or 1920, resize_height or 1080)
    if est_gb > 4.0:
        logger.warning(
            f"CẢNH BÁO RAM: cấu hình hiện tại (max_frames={max_frames}, "
            f"resize={resize_width}x{resize_height}) cần tối đa ~{est_gb:.1f}GB RAM "
            f"nếu video đủ dài. Nếu Kaggle bị kill kernel, giảm max_frames hoặc "
            f"resize_width/height xuống nữa."
        )

    container = av.open(video_path)
    stream = container.streams.video[0]
    fps_original = float(stream.average_rate) or 25.0
    stream.thread_type = "AUTO"

    total_frames_estimate = stream.frames or 0
    if total_frames_estimate == 0 and stream.duration and stream.time_base:
        total_frames_estimate = int(float(stream.duration * stream.time_base) * fps_original)

    frame_stride = 1
    if total_frames_estimate > max_frames:
        frame_stride = max(1, total_frames_estimate // max_frames)
        logger.warning(
            f"Video có ~{total_frames_estimate} frame, vượt max_frames={max_frames} "
            f"-> lấy mẫu CÁCH ĐỀU mỗi {frame_stride} frame để phủ TOÀN BỘ video "
            f"(không chỉ đoạn đầu). fps hiệu dụng: {fps_original/frame_stride:.2f} "
            f"(gốc: {fps_original:.2f})."
        )

    frames = []
    frame_idx = 0
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
            if frame_idx % frame_stride == 0:
                try:
                    img_rgb = frame.to_ndarray(format="rgb24")
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    if resize_width is not None and resize_height is not None:
                        img_bgr = cv2.resize(img_bgr, (resize_width, resize_height))
                except Exception as e:
                    logger.debug(f"Bỏ qua frame lỗi convert: {e}")
                    frame_idx += 1
                    continue
                frames.append(img_bgr)
                if len(frames) >= max_frames:
                    logger.warning(f"Đã đạt max_frames={max_frames}, dừng sớm (đã phủ hết video nếu ước tính đúng).")
                    break
            frame_idx += 1
        if len(frames) >= max_frames:
            break

    container.close()

    if not frames:
        raise ValueError(f"Không đọc được frame nào từ video: {video_path}")
    fps_effective = fps_original / frame_stride
    return np.stack(frames, axis=0), fps_effective


def get_video_frames(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resize_width: Optional[int] = DEFAULT_READ_WIDTH,
    resize_height: Optional[int] = DEFAULT_READ_HEIGHT,
) -> Tuple[np.ndarray, float]:
    """
    Entry point chính — trả về (frames_bgr (T,H,W,3) uint8, fps), có cache.

    QUAN TRỌNG: mặc định RESIZE xuống 480x270 và giới hạn 2000 frame — đã
    kiểm chứng KHÔNG tràn RAM trên video thật 1920x1080/32.714 frame (dataset
    AIC). Chỉ đổi resize_width/height=None hoặc tăng max_frames nếu bạn CHẮC
    CHẮN có đủ RAM (dùng estimate_ram_gb() để tự kiểm tra trước).
    """
    cache_key = f"{video_path}::{max_frames}::{resize_width}x{resize_height}"
    if cache_key in _cache:
        return _cache[cache_key]

    frames, fps = _read_all_frames_pyav(
        video_path, max_frames=max_frames,
        resize_width=resize_width, resize_height=resize_height,
    )

    if len(_cache) >= _CACHE_MAX_VIDEOS:
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]
        logger.debug(f"Giải phóng cache video cũ: {oldest_key}")

    _cache[cache_key] = (frames, fps)
    return frames, fps


def get_frame_range(
    video_path: str, start_frame: int, end_frame: int,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resize_width: Optional[int] = DEFAULT_READ_WIDTH,
    resize_height: Optional[int] = DEFAULT_READ_HEIGHT,
):
    """Trả về (indices, frames_bgr) cho khoảng [start_frame, end_frame) —
    CHỈ DÙNG khi start_frame/end_frame chắc chắn cùng hệ quy chiếu fps với
    video gốc (ví dụ HistogramSSIMDetector/AutoShotDetector). KHÔNG dùng với
    OmniShotCutDetector — xem get_frame_range_by_time() bên dưới."""
    frames, fps = get_video_frames(
        video_path, max_frames=max_frames, resize_width=resize_width, resize_height=resize_height,
    )
    total = frames.shape[0]
    s = max(0, min(start_frame, total))
    e = max(0, min(end_frame, total))
    if s >= e:
        return [], np.zeros((0, *frames.shape[1:]), dtype=frames.dtype)
    indices = list(range(s, e))
    return indices, frames[s:e]


def get_frame_range_by_time(
    video_path: str, start_time: float, end_time: float,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resize_width: Optional[int] = DEFAULT_READ_WIDTH,
    resize_height: Optional[int] = DEFAULT_READ_HEIGHT,
):
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
    frames, fps = get_video_frames(
        video_path, max_frames=max_frames, resize_width=resize_width, resize_height=resize_height,
    )
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
