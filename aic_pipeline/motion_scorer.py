"""
Tầng 2 — MOTION SCORING (optical flow, CPU, rẻ — xem giải thích đầy đủ trong
README / lượt trao đổi trước). Logic giữ nguyên vì đã kiểm chứng qua test.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .shot_detector import Shot, detect_shots


@dataclasses.dataclass
class MotionProfile:
    shot_id: int
    mean_flow_magnitude: float
    max_flow_magnitude: float
    flow_variance: float
    per_frame_magnitude: List[float] = dataclasses.field(default_factory=list)
    motion_class: str = "static"   # "static" | "moderate" | "dynamic"


def _compute_optical_flow_series(
    video_path: str,
    shot: Shot,
    resize_to: Tuple[int, int] = (160, 90),
    frame_stride: int = 2,
) -> List[float]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start_frame)

    magnitudes: List[float] = []
    prev_gray = None
    frame_pos = shot.start_frame

    while frame_pos < shot.end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_pos - shot.start_frame) % frame_stride == 0:
            gray = cv2.cvtColor(cv2.resize(frame, resize_to), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes.append(float(mag.mean()))
            prev_gray = gray
        frame_pos += 1
    cap.release()
    return magnitudes


def score_motion(
    video_path: str,
    shots: List[Shot],
    static_threshold: float = 0.5,
    dynamic_threshold: float = 2.0,
    resize_to: Tuple[int, int] = (160, 90),
    frame_stride: int = 2,
) -> List[MotionProfile]:
    """Entry point Tầng 2. Xem calibrate_thresholds() để tự tìm ngưỡng trên data thật."""
    profiles: List[MotionProfile] = []
    for shot in shots:
        mags = _compute_optical_flow_series(video_path, shot, resize_to, frame_stride)
        if not mags:
            mags = [0.0]
        mags_arr = np.array(mags)
        mean_mag = float(mags_arr.mean())
        motion_class = (
            "static" if mean_mag < static_threshold
            else "dynamic" if mean_mag > dynamic_threshold
            else "moderate"
        )
        profiles.append(
            MotionProfile(
                shot_id=shot.shot_id,
                mean_flow_magnitude=mean_mag,
                max_flow_magnitude=float(mags_arr.max()),
                flow_variance=float(mags_arr.var()),
                per_frame_magnitude=mags,
                motion_class=motion_class,
            )
        )
    return profiles


def calibrate_thresholds(
    video_paths: List[str],
    shots_per_video: List[List[Shot]],
    low_percentile: float = 33.0,
    high_percentile: float = 66.0,
    resize_to: Tuple[int, int] = (160, 90),
    frame_stride: int = 2,
) -> Tuple[float, float]:
    """
    Chạy trên tập video mẫu (data tự crawl của bạn) để tìm ngưỡng phù hợp domain,
    thay vì dùng số mặc định đoán sẵn.

        videos = glob.glob("data_crawl/*.mp4")
        all_shots = [detect_shots(v) for v in videos]
        lo, hi = calibrate_thresholds(videos, all_shots)
    """
    all_means: List[float] = []
    for video_path, shots in zip(video_paths, shots_per_video):
        for shot in shots:
            mags = _compute_optical_flow_series(video_path, shot, resize_to, frame_stride)
            if mags:
                all_means.append(float(np.mean(mags)))
    if not all_means:
        raise ValueError("Không tính được motion trên tập mẫu — kiểm tra lại video_paths/shots.")
    arr = np.array(all_means)
    lo = float(np.percentile(arr, low_percentile))
    hi = float(np.percentile(arr, high_percentile))
    return lo, hi


def auto_calibrate_thresholds_gmm(
    video_paths: List[str],
    shots_per_video: Optional[List[List[Shot]]] = None,
    n_components: int = 3,
    resize_to: Tuple[int, int] = (160, 90),
    frame_stride: int = 2,
    random_state: int = 42,
) -> Tuple[float, float, dict]:
    """
    VÁ NHƯỢC ĐIỂM #1: tự tìm ngưỡng static/dynamic bằng Gaussian Mixture Model
    (3 thành phần: static/moderate/dynamic), KHÔNG cần bạn tự chọn percentile
    thủ công (33%/66% là số đoán, không phản ánh đúng phân bố thật của domain).

    Khác calibrate_thresholds() (percentile cố định — LUÔN chia 33%/33%/34%
    dù phân bố thực tế lệch hẳn về 1 phía), hàm này FIT một Gaussian Mixture
    3 thành phần lên phân bố motion score thật, rồi lấy ranh giới giữa các
    cụm làm ngưỡng — tự thích nghi với hình dạng phân bố thật (ví dụ domain
    tin tức có thể có rất nhiều shot tĩnh và ít shot động, GMM sẽ tự phản ánh
    đúng tỉ lệ đó thay vì ép về 33/33/34).

    Args:
        video_paths: danh sách video mẫu (khuyến nghị 15-30 video đại diện).
        shots_per_video: nếu None, tự gọi detect_shots() cho từng video.
        n_components: số cụm GMM, mặc định 3 (static/moderate/dynamic) khớp
                      đúng 3 motion_class hiện có trong pipeline.

    Returns:
        (static_threshold, dynamic_threshold, debug_info) — debug_info chứa
        thêm thông tin phân bố (means, weights của từng cụm) để bạn kiểm tra
        bằng mắt trước khi dùng cho production.
    """
    from sklearn.mixture import GaussianMixture

    if shots_per_video is None:
        shots_per_video = [detect_shots(v) for v in video_paths]

    all_means: List[float] = []
    for video_path, shots in zip(video_paths, shots_per_video):
        for shot in shots:
            mags = _compute_optical_flow_series(video_path, shot, resize_to, frame_stride)
            if mags:
                all_means.append(float(np.mean(mags)))

    if len(all_means) < n_components * 3:
        raise ValueError(
            f"Chỉ có {len(all_means)} shot mẫu — cần ít nhất {n_components * 3} "
            f"để fit GMM đáng tin cậy. Thêm video mẫu hoặc dùng calibrate_thresholds() "
            f"(percentile đơn giản) thay thế."
        )

    X = np.array(all_means).reshape(-1, 1)
    # log-transform vì motion score luôn dương và thường lệch phải (nhiều shot
    # tĩnh, ít shot cực động) — GMM fit tốt hơn trên phân bố gần chuẩn hơn là
    # trên phân bố lệch mạnh.
    X_log = np.log1p(X)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=5)
    gmm.fit(X_log)

    # sắp xếp các cụm theo mean tăng dần: cụm 0 = static, cụm giữa = moderate, cụm cuối = dynamic
    means_log = gmm.means_.flatten()
    order = np.argsort(means_log)
    sorted_means = np.expm1(means_log[order])
    weights_sorted = gmm.weights_[order]

    # ngưỡng = điểm giữa (trong không gian log) giữa 2 cụm liền kề, chuyển ngược lại thang gốc
    static_threshold = float(np.expm1((means_log[order[0]] + means_log[order[1]]) / 2))
    dynamic_threshold = float(np.expm1((means_log[order[-2]] + means_log[order[-1]]) / 2))

    debug_info = {
        "n_samples": len(all_means),
        "cluster_means_sorted": sorted_means.tolist(),
        "cluster_weights_sorted": weights_sorted.tolist(),
        "raw_min": float(X.min()), "raw_max": float(X.max()),
        "raw_median": float(np.median(X)),
    }

    return static_threshold, dynamic_threshold, debug_info
