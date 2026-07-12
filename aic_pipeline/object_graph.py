"""
OBJECT GRAPH (MỚI) — VÁ NHƯỢC ĐIỂM: Temporal Reranker (Tầng 5) trước đây chỉ
nối SHOT-VỚI-SHOT theo thời gian, chưa mô hình hoá tương quan giữa các ĐỐI
TƯỢNG bên trong 1 shot.

Thiết kế 2 tầng, có GPU-fallback rõ ràng:
  - Nếu có model object detection (YOLO qua ultralytics, hoặc bạn tự cắm
    RF-DETR/RAM++ như các paper đã khảo sát) -> dùng để detect object mỗi
    N frame trong shot.
  - Nếu KHÔNG có -> fallback dùng contour detection (OpenCV thuần, không cần
    GPU/model tải về) làm "proxy" phát hiện vùng chuyển động — kém chính xác
    hơn YOLO thật nhưng KHÔNG BAO GIỜ crash vì thiếu model.

Sau khi có object mỗi frame, dùng centroid tracking đơn giản để gán ID xuyên
suốt shot. Kết quả: ObjectGraph — node là object-track, cạnh là quan hệ
không gian-thời gian giữa chúng. Dùng bổ sung cho Temporal Reranker qua
StageHit.meta khi truy vấn cần nhiều đối tượng tương tác.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .shot_detector import Shot

logger = logging.getLogger("aic_pipeline.object_graph")


@dataclasses.dataclass
class DetectedObject:
    frame_index: int
    bbox: Tuple[int, int, int, int]
    label: str = "object"
    confidence: float = 1.0

    @property
    def centroid(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


@dataclasses.dataclass
class ObjectTrack:
    track_id: int
    detections: List[DetectedObject] = dataclasses.field(default_factory=list)
    label: str = "object"

    @property
    def first_seen(self) -> int:
        return self.detections[0].frame_index if self.detections else -1

    @property
    def last_seen(self) -> int:
        return self.detections[-1].frame_index if self.detections else -1

    def centroid_at(self, frame_index: int) -> Optional[Tuple[float, float]]:
        for d in self.detections:
            if d.frame_index == frame_index:
                return d.centroid
        return None


@dataclasses.dataclass
class ObjectGraphEdge:
    track_id_a: int
    track_id_b: int
    co_occurrence_frames: int
    mean_distance: float
    interaction_score: float


@dataclasses.dataclass
class ObjectGraph:
    shot_id: int
    tracks: List[ObjectTrack]
    edges: List[ObjectGraphEdge]

    def get_track(self, track_id: int) -> Optional[ObjectTrack]:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None


class ObjectDetectorBackend(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> List[DetectedObject]: ...


class ContourFallbackDetector:
    """
    Fallback KHÔNG CẦN model tải về, KHÔNG CẦN GPU — background subtraction
    + contour làm proxy. Đảm bảo ObjectGraph LUÔN CHẠY ĐƯỢC dù chưa cài
    detector nặng — cùng nguyên tắc "sàn an toàn" như GPUShotDetector.
    """

    def __init__(self, min_area: int = 200, max_objects: int = 15):
        self.min_area = min_area
        self.max_objects = max_objects
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        fg_mask = self._bg_subtractor.apply(frame_bgr)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            objects.append(DetectedObject(frame_index=-1, bbox=(x, y, w, h), label="motion_region"))

        objects.sort(key=lambda o: o.bbox[2] * o.bbox[3], reverse=True)
        return objects[: self.max_objects]


class YoloDetector:
    """
    Wrapper YOLO qua ultralytics — cần: !pip install -q ultralytics
    Khuyến nghị dùng thay ContourFallbackDetector khi có GPU, độ chính xác
    cao hơn nhiều và có label ý nghĩa (person, car...).
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cuda", conf_threshold: float = 0.35):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("YoloDetector cần ultralytics: !pip install -q ultralytics")
        self._model = YOLO(self.model_name)

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        self._lazy_load()
        results = self._model.predict(
            frame_bgr, device=self.device, conf=self.conf_threshold, verbose=False,
        )
        objects = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                label = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                objects.append(
                    DetectedObject(
                        frame_index=-1, bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        label=label, confidence=conf,
                    )
                )
        return objects


def _track_objects_in_shot(
    detections_per_frame: Dict[int, List[DetectedObject]],
    max_match_distance: float = 80.0,
) -> List[ObjectTrack]:
    """Centroid tracking đơn giản, đủ tin cậy trong phạm vi 1 shot ngắn."""
    tracks: List[ObjectTrack] = []
    active_track_last_centroid: Dict[int, Tuple[float, float]] = {}
    next_track_id = 0

    for frame_idx in sorted(detections_per_frame.keys()):
        dets = detections_per_frame[frame_idx]
        for d in dets:
            d.frame_index = frame_idx

        unmatched = list(range(len(dets)))
        for track_id, last_c in list(active_track_last_centroid.items()):
            if not unmatched:
                break
            best_i, best_dist = None, max_match_distance
            for i in unmatched:
                c = dets[i].centroid
                dist = float(np.hypot(c[0] - last_c[0], c[1] - last_c[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            if best_i is not None:
                track = next(t for t in tracks if t.track_id == track_id)
                track.detections.append(dets[best_i])
                active_track_last_centroid[track_id] = dets[best_i].centroid
                unmatched.remove(best_i)

        for i in unmatched:
            new_track = ObjectTrack(track_id=next_track_id, detections=[dets[i]], label=dets[i].label)
            tracks.append(new_track)
            active_track_last_centroid[next_track_id] = dets[i].centroid
            next_track_id += 1

    return tracks


def build_object_graph(
    video_path: str,
    shot: Shot,
    detector: Optional[ObjectDetectorBackend] = None,
    frame_stride: int = 5,
    max_match_distance: float = 80.0,
    interaction_distance_threshold: float = 150.0,
) -> ObjectGraph:
    """
    Entry point: xây ObjectGraph cho 1 shot.

    Args:
        detector: None -> ContourFallbackDetector (không cần GPU). Truyền
                  YoloDetector(device="cuda") để dùng model thật.
        frame_stride: chạy detector cách nhau vài frame (đắt hơn optical flow).
        interaction_distance_threshold: ngưỡng khoảng cách (pixel, ảnh gốc)
                  để coi 2 track là "tương tác".
    """
    detector = detector or ContourFallbackDetector()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start_frame)

    detections_per_frame: Dict[int, List[DetectedObject]] = {}
    frame_pos = shot.start_frame
    while frame_pos < shot.end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_pos - shot.start_frame) % frame_stride == 0:
            dets = detector.detect(frame)
            detections_per_frame[frame_pos] = dets
        frame_pos += 1
    cap.release()

    tracks = _track_objects_in_shot(detections_per_frame, max_match_distance)

    edges: List[ObjectGraphEdge] = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            t_a, t_b = tracks[i], tracks[j]
            common_frames = set(d.frame_index for d in t_a.detections) & set(
                d.frame_index for d in t_b.detections
            )
            if not common_frames:
                continue
            distances = []
            for f in common_frames:
                ca, cb = t_a.centroid_at(f), t_b.centroid_at(f)
                if ca is not None and cb is not None:
                    distances.append(float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])))
            if not distances:
                continue
            mean_dist = float(np.mean(distances))
            if mean_dist > interaction_distance_threshold:
                continue

            proximity_score = max(0.0, 1.0 - mean_dist / interaction_distance_threshold)
            co_occurrence_score = len(common_frames) / max(1, len(detections_per_frame))
            interaction_score = 0.6 * proximity_score + 0.4 * co_occurrence_score

            edges.append(
                ObjectGraphEdge(
                    track_id_a=t_a.track_id, track_id_b=t_b.track_id,
                    co_occurrence_frames=len(common_frames), mean_distance=mean_dist,
                    interaction_score=interaction_score,
                )
            )

    return ObjectGraph(shot_id=shot.shot_id, tracks=tracks, edges=edges)


def summarize_graph_for_query(graph: ObjectGraph, top_k_edges: int = 5) -> dict:
    """Tóm tắt ObjectGraph thành dict gọn, nhét vào StageHit.meta của Tầng 5."""
    top_edges = sorted(graph.edges, key=lambda e: -e.interaction_score)[:top_k_edges]
    return {
        "n_tracks": len(graph.tracks),
        "n_interactions": len(graph.edges),
        "top_interactions": [
            {
                "labels": (
                    graph.get_track(e.track_id_a).label,
                    graph.get_track(e.track_id_b).label,
                ),
                "interaction_score": round(e.interaction_score, 3),
                "co_occurrence_frames": e.co_occurrence_frames,
            }
            for e in top_edges
        ],
    }
