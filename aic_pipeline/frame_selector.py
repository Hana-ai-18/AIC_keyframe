"""
Tầng 4 — FRAME SELECTION (2 chế độ: "cheap" và "semantic")
=============================================================

Bản này SỬA lỗ hổng đã chỉ ra khi so sánh với NII-UIT: bản trước chỉ dùng đặc
trưng thị giác thấp (màu/cạnh) để đo đa dạng, có thể bỏ sót khác biệt NGỮ NGHĨA
(2 frame giống nhau về bố cục màu nhưng khác nội dung). Vì bạn có A100, giờ có
chế độ "semantic" dùng CLIP embedding thật.

3 bước (không đổi so với bản trước), chỉ đổi cách trích đặc trưng ở bước 2:

  Bước 1 — QUALITY FILTER: loại frame quá mờ (Laplacian variance) hoặc phơi
           sáng xấu, TRƯỚC khi đưa vào bước chọn.

  Bước 2 — FEATURE EXTRACTION — 2 chế độ:
      mode="cheap"    : color histogram + edge density theo lưới (không cần GPU,
                         chạy được ở mọi máy, dùng khi feature_mode chưa cấu hình
                         hoặc không có torch).
      mode="semantic" : CLIP embedding (qua embeddings.ClipEmbedder) — bắt được
                         khác biệt ngữ nghĩa thật, mạnh hơn NII-UIT (NII-UIT dùng
                         BEiT-3 tương đương ý tưởng, ở đây dùng CLIP làm mặc định
                         vì nhẹ, tải nhanh trên Kaggle; có thể thay bằng SigLIP2
                         nếu muốn mạnh hơn nữa).

  Bước 3 — DIVERSITY SELECTION: farthest-point-sampling có trọng số chất lượng,
           logic KHÔNG ĐỔI so với bản trước (đã kiểm chứng bằng test).
"""
from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional

import cv2
import numpy as np

from .shot_detector import Shot

logger = logging.getLogger("aic_pipeline.frame_selector")


@dataclasses.dataclass
class Keyframe:
    shot_id: int
    frame_index: int
    timestamp: float
    sharpness: float
    diversity_rank: int
    feature_mode: str = "cheap"     # "cheap" | "semantic" — để biết keyframe này chọn bằng cách nào
    image: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None   # lưu lại embedding CLIP nếu mode=semantic (tái dùng cho index)


def _laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_penalty(gray: np.ndarray) -> float:
    mean_brightness = float(gray.mean())
    return abs(mean_brightness - 128.0) / 128.0


def _color_edge_feature(frame_bgr: np.ndarray, resize_to=(64, 36)) -> np.ndarray:
    """Chế độ 'cheap' — không đổi so với bản trước."""
    small = cv2.resize(frame_bgr, resize_to)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    grid = []
    for i in range(4):
        for j in range(4):
            cell = edges[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            grid.append(cell.mean() / 255.0)
    return np.concatenate([hist.flatten(), np.array(grid)])


def _farthest_point_sampling(
    features: np.ndarray,
    quality_scores: np.ndarray,
    n_select: int,
    quality_weight: float = 0.3,
) -> List[int]:
    """Không đổi so với bản trước — đã kiểm chứng qua test."""
    n = features.shape[0]
    if n <= n_select:
        return list(range(n))

    q_norm = (quality_scores - quality_scores.min()) / (
        quality_scores.max() - quality_scores.min() + 1e-8
    )

    selected = [int(np.argmax(quality_scores))]
    min_dists = np.linalg.norm(features - features[selected[0]], axis=1)

    while len(selected) < n_select:
        dist_norm = min_dists / (min_dists.max() + 1e-8)
        score = (1 - quality_weight) * dist_norm + quality_weight * q_norm
        for s in selected:
            score[s] = -1
        next_idx = int(np.argmax(score))
        selected.append(next_idx)
        new_dists = np.linalg.norm(features - features[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return selected


def select_keyframes(
    video_path: str,
    shot: Shot,
    n_keyframes: int,
    feature_mode: str = "cheap",
    embedder=None,   # instance của embeddings.ClipEmbedder, bắt buộc nếu feature_mode="semantic"
    quality_weight: float = 0.3,
    min_sharpness_percentile: float = 15.0,
    frame_stride: int = 1,
    store_images: bool = True,
    store_embeddings: bool = False,
) -> List[Keyframe]:
    """
    Entry point Tầng 4.

    Args:
        feature_mode: "cheap" (mặc định, không cần GPU) hoặc "semantic" (cần
                      truyền `embedder`, dùng CLIP thật — khuyến nghị khi có A100).
        embedder: instance ClipEmbedder — BẮT BUỘC nếu feature_mode="semantic".
        store_embeddings: True thì lưu vector CLIP vào Keyframe.embedding, để
                          TÁI DÙNG cho bước index retrieval chính thức sau này
                          (tránh phải encode lại lần 2).
        (các tham số khác giống bản trước, không đổi ý nghĩa)
    """
    if feature_mode == "semantic" and embedder is None:
        raise ValueError(
            "feature_mode='semantic' yêu cầu truyền embedder (ClipEmbedder). "
            "Ví dụ: from aic_pipeline.embeddings import ClipEmbedder; "
            "embedder = ClipEmbedder(device='cuda')"
        )
    if feature_mode not in ("cheap", "semantic"):
        raise ValueError(f"feature_mode phải là 'cheap' hoặc 'semantic', được: {feature_mode}")

    # ĐÃ VÁ BUG NGHIÊM TRỌNG: trước đây dùng cv2.VideoCapture — KHÔNG hỗ trợ
    # codec AV1 (dataset AIC dùng AV1), khiến cap.read() trả ret=False ngay
    # từ frame đầu, hàm luôn trả về [] (0 keyframe) dù shot hợp lệ. Dùng
    # video_reader (PyAV, đã hỗ trợ AV1) thay thế.
    #
    # DÙNG start_time/end_time (giây, LUÔN ĐÚNG) thay vì start_frame/end_frame
    # (có thể sai hệ quy chiếu nếu detector — như OmniShotCutDetector — tính
    # frame theo fps đã subsample khác fps gốc của video_reader).
    from .video_reader import get_frame_range_by_time, get_video_frames

    frame_indices_range, frames_bgr = get_frame_range_by_time(video_path, shot.start_time, shot.end_time)
    _all_frames, fps = get_video_frames(video_path)

    candidates_idx: List[int] = []
    candidates_img: List[np.ndarray] = []
    sharpness_list: List[float] = []
    exposure_list: List[float] = []

    range_start = frame_indices_range[0] if frame_indices_range else 0
    for local_i, global_frame_idx in enumerate(frame_indices_range):
        if (global_frame_idx - range_start) % frame_stride == 0:
            frame = frames_bgr[local_i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            candidates_idx.append(global_frame_idx)
            candidates_img.append(frame)
            sharpness_list.append(_laplacian_sharpness(gray))
            exposure_list.append(_exposure_penalty(gray))

    if not candidates_idx:
        return []

    sharpness_arr = np.array(sharpness_list)
    exposure_arr = np.array(exposure_list)
    quality_scores = sharpness_arr * (1.0 - exposure_arr)

    if len(candidates_idx) > n_keyframes:
        threshold = np.percentile(quality_scores, min_sharpness_percentile)
        keep_mask = quality_scores >= threshold
        if keep_mask.sum() < n_keyframes:
            keep_mask = np.ones_like(keep_mask, dtype=bool)
    else:
        keep_mask = np.ones(len(candidates_idx), dtype=bool)

    kept_idx = [i for i, k in enumerate(keep_mask) if k]
    kept_images = [candidates_img[i] for i in kept_idx]
    kept_quality = quality_scores[kept_idx]

    # --- Bước 2: trích đặc trưng theo feature_mode ---
    all_embeddings: Optional[np.ndarray] = None
    if feature_mode == "cheap":
        features = np.stack([_color_edge_feature(img) for img in kept_images])
    else:  # semantic
        all_embeddings = embedder.encode_images(kept_images)
        features = all_embeddings

    n_select = min(n_keyframes, len(kept_idx))
    selected_local = _farthest_point_sampling(features, kept_quality, n_select, quality_weight)

    results: List[Keyframe] = []
    for rank, local_i in enumerate(selected_local):
        global_i = kept_idx[local_i]
        results.append(
            Keyframe(
                shot_id=shot.shot_id,
                frame_index=candidates_idx[global_i],
                timestamp=candidates_idx[global_i] / fps,
                sharpness=float(sharpness_arr[global_i]),
                diversity_rank=rank,
                feature_mode=feature_mode,
                image=candidates_img[global_i] if store_images else None,
                embedding=(
                    all_embeddings[local_i]
                    if (store_embeddings and all_embeddings is not None) else None
                ),
            )
        )
    results.sort(key=lambda k: k.timestamp)
    return results
