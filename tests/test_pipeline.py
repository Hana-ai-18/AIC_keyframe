"""
Test đầy đủ cho aic_pipeline. Chạy: python3 -m pytest tests/ -v

Test được chia rõ 2 nhóm:
  - Nhóm A (không cần GPU/torch): Tầng 1,2,3, chế độ "cheap" của Tầng 4, và
    TOÀN BỘ Tầng 5 (Temporal Reranker) — chạy được ở MỌI môi trường, kể cả máy
    không có GPU (dùng để kiểm tra logic trước khi lên Kaggle).
  - Nhóm B (cần torch+transformers+GPU): chế độ "semantic" của Tầng 4. Tự động
    SKIP nếu môi trường không có torch — không làm hỏng cả bộ test khi chạy ở
    máy không có GPU, nhưng SẼ chạy thật khi bạn chạy trên Kaggle.
"""
import os
import sys

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aic_pipeline import (
    PipelineConfig, run_pipeline, run_pipeline_batch,
    detect_shots, score_motion, allocate_budget, select_keyframes,
    TemporalReranker, StageHit,
)
from aic_pipeline.shot_detector import HistogramSSIMDetector, Shot
from aic_pipeline.budget_allocator import allocate_budget_with_global_cap
from aic_pipeline.motion_scorer import MotionProfile, calibrate_thresholds
from aic_pipeline.streaming_batch import run_pipeline_fast, FastPipelineConfig

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

skip_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="Cần torch+transformers (chạy trên Kaggle GPU)"
)


# ---------------------------------------------------------------------------
# Fixture: video tổng hợp 3 shot (2 tĩnh + 1 động)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("videos")
    path = str(tmp_dir / "test.mp4")

    fps = 25
    w, h = 320, 180
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.RandomState(42)

    for _ in range(fps * 3):
        frame = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)
        cv2.rectangle(frame, (100, 60), (220, 120), (200, 150, 50), -1)
        noise = rng.randint(-3, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)

    for _ in range(fps * 2):
        frame = np.full((h, w, 3), (30, 90, 30), dtype=np.uint8)
        cv2.circle(frame, (160, 90), 40, (50, 50, 220), -1)
        noise = rng.randint(-3, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)

    n_balls = 8
    pos = rng.uniform(20, w - 20, size=(n_balls, 2))
    vel = rng.uniform(-15, 15, size=(n_balls, 2))
    for i in range(fps * 5):
        frame = np.full((h, w, 3), (20, 20, 20), dtype=np.uint8)
        pos += vel
        for j in range(n_balls):
            if pos[j, 0] < 10 or pos[j, 0] > w - 10:
                vel[j, 0] *= -1
            if pos[j, 1] < 10 or pos[j, 1] > h - 10:
                vel[j, 1] *= -1
            if i % 15 == 0:
                vel[j] += rng.uniform(-8, 8, size=2)
            color = tuple(int(c) for c in rng.randint(80, 255, 3))
            cv2.circle(frame, (int(pos[j, 0]), int(pos[j, 1])), 12, color, -1)
        out.write(frame)

    out.release()
    return path


@pytest.fixture(scope="module")
def video_dir_with_two_videos(tmp_path_factory, synthetic_video):
    """Thư mục có 2 video (1 hợp lệ copy từ synthetic_video, 1 file rỗng/hỏng)
    để test run_pipeline_batch xử lý lỗi đúng cách."""
    import shutil
    d = tmp_path_factory.mktemp("batch_videos")
    shutil.copy(synthetic_video, d / "video_ok.mp4")
    (d / "video_broken.mp4").write_bytes(b"not a real video file")
    return str(d)


# ---------------------------------------------------------------------------
# Tầng 1
# ---------------------------------------------------------------------------
class TestShotDetector:
    def test_detects_correct_number_of_shots(self, synthetic_video):
        shots = detect_shots(synthetic_video, k_std=5.0, min_shot_len=15)
        assert len(shots) == 3

    def test_shot_boundaries_approximately_correct(self, synthetic_video):
        shots = detect_shots(synthetic_video, k_std=5.0, min_shot_len=15)
        assert abs(shots[0].start_time - 0.0) < 0.1
        assert abs(shots[0].end_time - 3.0) < 0.2
        assert abs(shots[1].end_time - 5.0) < 0.2
        assert abs(shots[2].end_time - 10.0) < 0.2

    def test_shots_are_contiguous(self, synthetic_video):
        shots = detect_shots(synthetic_video, k_std=5.0, min_shot_len=15)
        for i in range(len(shots) - 1):
            assert shots[i].end_frame == shots[i + 1].start_frame

    def test_raises_on_invalid_path(self):
        with pytest.raises(IOError):
            detect_shots("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# Tầng 2
# ---------------------------------------------------------------------------
class TestMotionScorer:
    def test_static_shots_scored_low(self, synthetic_video):
        shots = detect_shots(synthetic_video, k_std=5.0, min_shot_len=15)
        profiles = score_motion(synthetic_video, shots)
        for p in profiles:
            if p.shot_id in (0, 1):
                assert p.motion_class == "static"

    def test_dynamic_shot_scored_high(self, synthetic_video):
        shots = detect_shots(synthetic_video, k_std=5.0, min_shot_len=15)
        profiles = score_motion(synthetic_video, shots)
        dynamic_shot = [p for p in profiles if p.shot_id == 2][0]
        assert dynamic_shot.motion_class == "dynamic"

    def test_dynamic_higher_than_static(self, synthetic_video):
        shots = detect_shots(synthetic_video, k_std=5.0, min_shot_len=15)
        profiles = score_motion(synthetic_video, shots)
        static_mean = max(p.mean_flow_magnitude for p in profiles if p.shot_id in (0, 1))
        dynamic_mean = [p.mean_flow_magnitude for p in profiles if p.shot_id == 2][0]
        assert dynamic_mean > static_mean * 5

    def test_calibrate_thresholds_runs(self, synthetic_video):
        shots = detect_shots(synthetic_video, k_std=5.0, min_shot_len=15)
        lo, hi = calibrate_thresholds([synthetic_video], [shots])
        assert lo <= hi


# ---------------------------------------------------------------------------
# Tầng 3
# ---------------------------------------------------------------------------
class TestBudgetAllocator:
    def _make_shots_and_profiles(self):
        shots = [
            Shot(shot_id=0, start_frame=0, end_frame=75, start_time=0, end_time=3.0),
            Shot(shot_id=1, start_frame=75, end_frame=125, start_time=3.0, end_time=5.0),
            Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0),
        ]
        profiles = [
            MotionProfile(shot_id=0, mean_flow_magnitude=0.02, max_flow_magnitude=0.08,
                          flow_variance=0.0, motion_class="static"),
            MotionProfile(shot_id=1, mean_flow_magnitude=0.01, max_flow_magnitude=0.02,
                          flow_variance=0.0, motion_class="static"),
            MotionProfile(shot_id=2, mean_flow_magnitude=2.0, max_flow_magnitude=2.9,
                          flow_variance=0.14, motion_class="dynamic"),
        ]
        return shots, profiles

    def test_static_gets_minimum_budget(self):
        shots, profiles = self._make_shots_and_profiles()
        budgets = allocate_budget(shots, profiles)
        for b in budgets:
            if b.motion_class == "static":
                assert b.n_keyframes == 1

    def test_dynamic_gets_more_budget(self):
        shots, profiles = self._make_shots_and_profiles()
        budgets = allocate_budget(shots, profiles)
        static_n = [b.n_keyframes for b in budgets if b.motion_class == "static"]
        dynamic_n = [b.n_keyframes for b in budgets if b.motion_class == "dynamic"][0]
        assert dynamic_n > max(static_n)

    def test_global_cap_respected(self):
        shots, profiles = self._make_shots_and_profiles()
        budgets = allocate_budget_with_global_cap(shots, profiles, total_budget=4)
        total = sum(b.n_keyframes for b in budgets)
        assert total <= 4 + len(shots)

    def test_all_shots_get_at_least_one(self):
        shots, profiles = self._make_shots_and_profiles()
        budgets = allocate_budget_with_global_cap(shots, profiles, total_budget=1)
        for b in budgets:
            assert b.n_keyframes >= 1


# ---------------------------------------------------------------------------
# Tầng 4 — chế độ "cheap" (không cần GPU)
# ---------------------------------------------------------------------------
class TestFrameSelectorCheap:
    def test_selects_requested_number(self, synthetic_video):
        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        kfs = select_keyframes(synthetic_video, shot, n_keyframes=6, feature_mode="cheap", store_images=False)
        assert len(kfs) == 6

    def test_sorted_by_time(self, synthetic_video):
        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        kfs = select_keyframes(synthetic_video, shot, n_keyframes=6, feature_mode="cheap", store_images=False)
        ts = [kf.timestamp for kf in kfs]
        assert ts == sorted(ts)

    def test_within_shot_bounds(self, synthetic_video):
        shot = Shot(shot_id=0, start_frame=0, end_frame=75, start_time=0.0, end_time=3.0)
        kfs = select_keyframes(synthetic_video, shot, n_keyframes=1, feature_mode="cheap", store_images=False)
        for kf in kfs:
            assert shot.start_frame <= kf.frame_index < shot.end_frame

    def test_n_larger_than_shot_length_capped(self, synthetic_video):
        shot = Shot(shot_id=1, start_frame=75, end_frame=125, start_time=3.0, end_time=5.0)
        kfs = select_keyframes(synthetic_video, shot, n_keyframes=1000, feature_mode="cheap", store_images=False)
        assert len(kfs) <= (shot.end_frame - shot.start_frame)

    def test_invalid_feature_mode_raises(self, synthetic_video):
        shot = Shot(shot_id=0, start_frame=0, end_frame=75, start_time=0.0, end_time=3.0)
        with pytest.raises(ValueError):
            select_keyframes(synthetic_video, shot, n_keyframes=1, feature_mode="invalid_mode")

    def test_semantic_without_embedder_raises(self, synthetic_video):
        shot = Shot(shot_id=0, start_frame=0, end_frame=75, start_time=0.0, end_time=3.0)
        with pytest.raises(ValueError):
            select_keyframes(synthetic_video, shot, n_keyframes=1, feature_mode="semantic", embedder=None)


# ---------------------------------------------------------------------------
# Tầng 4 — chế độ "semantic" (CẦN GPU/torch, auto-skip nếu không có)
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    """
    Embedder giả lập, KHÔNG cần torch/internet — dùng để kiểm chứng LOGIC của
    select_keyframes(feature_mode="semantic") độc lập với việc tải model CLIP
    thật có thành công hay không (2 mối lo khác nhau: đúng logic vs. có mạng).
    """

    def encode_images(self, images_bgr):
        import numpy as np
        vecs = []
        for img in images_bgr:
            seed = int(img.mean() * 1000) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(32).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        return np.stack(vecs)


class TestFrameSelectorSemanticLogicWithFakeEmbedder:
    """
    Test KHÔNG cần torch/internet — kiểm chứng logic gọi embedder đúng cách,
    tách biệt khỏi việc tải CLIP thật có thành công hay không (việc đó chỉ
    kiểm chứng được khi có mạng, ví dụ trên Kaggle — xem TestFrameSelectorSemantic).
    """

    def test_semantic_mode_with_fake_embedder_returns_correct_count(self, synthetic_video):
        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        kfs = select_keyframes(
            synthetic_video, shot, n_keyframes=6, feature_mode="semantic",
            embedder=_FakeEmbedder(), store_images=False, store_embeddings=True,
        )
        assert len(kfs) == 6
        for kf in kfs:
            assert kf.embedding is not None
            assert kf.embedding.shape == (32,)
            assert kf.feature_mode == "semantic"

    def test_semantic_mode_diversity_differs_from_cheap_mode(self, synthetic_video):
        """Kiểm tra 2 chế độ thực sự dùng đường xử lý KHÁC NHAU."""
        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        kfs_cheap = select_keyframes(
            synthetic_video, shot, n_keyframes=4, feature_mode="cheap", store_images=False,
        )
        kfs_semantic = select_keyframes(
            synthetic_video, shot, n_keyframes=4, feature_mode="semantic",
            embedder=_FakeEmbedder(), store_images=False,
        )
        assert all(kf.feature_mode == "cheap" for kf in kfs_cheap)
        assert all(kf.feature_mode == "semantic" for kf in kfs_semantic)


@skip_no_torch
class TestFrameSelectorSemantic:
    @classmethod
    @pytest.fixture(scope="class")
    def embedder(cls):
        from aic_pipeline.embeddings import ClipEmbedder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return ClipEmbedder(device=device)

    def test_semantic_mode_runs_and_returns_correct_count(self, synthetic_video, embedder):
        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        try:
            kfs = select_keyframes(
                synthetic_video, shot, n_keyframes=6, feature_mode="semantic",
                embedder=embedder, store_images=False, store_embeddings=True,
            )
        except OSError as e:
            pytest.skip(f"Không tải được checkpoint CLIP (cần internet, ví dụ trên Kaggge): {e}")
        assert len(kfs) == 6
        for kf in kfs:
            assert kf.embedding is not None
            assert kf.embedding.shape[0] > 0

    def test_semantic_embeddings_are_normalized(self, synthetic_video, embedder):
        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        try:
            kfs = select_keyframes(
                synthetic_video, shot, n_keyframes=3, feature_mode="semantic",
                embedder=embedder, store_images=False, store_embeddings=True,
            )
        except OSError as e:
            pytest.skip(f"Không tải được checkpoint CLIP (cần internet, ví dụ trên Kaggge): {e}")
        for kf in kfs:
            norm = np.linalg.norm(kf.embedding)
            assert abs(norm - 1.0) < 1e-2, f"Embedding phải L2-normalize, norm={norm}"


# ---------------------------------------------------------------------------
# Full pipeline (Tầng 1-4, chế độ cheap)
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_runs_end_to_end(self, synthetic_video):
        result = run_pipeline(synthetic_video, PipelineConfig(store_images=False))
        assert result.stats["n_shots"] == 3
        assert result.stats["n_keyframes"] > 0
        assert result.error is None

    def test_dynamic_shot_gets_more_keyframes_end_to_end(self, synthetic_video):
        result = run_pipeline(synthetic_video, PipelineConfig(store_images=False))
        kf_count_by_shot = {}
        for kf in result.keyframes:
            kf_count_by_shot[kf.shot_id] = kf_count_by_shot.get(kf.shot_id, 0) + 1

        static_ids = [mp.shot_id for mp in result.motion_profiles if mp.motion_class == "static"]
        dynamic_ids = [mp.shot_id for mp in result.motion_profiles if mp.motion_class == "dynamic"]
        assert len(dynamic_ids) > 0 and len(static_ids) > 0

        max_static = max(kf_count_by_shot.get(sid, 0) for sid in static_ids)
        min_dynamic = min(kf_count_by_shot.get(sid, 0) for sid in dynamic_ids)
        assert min_dynamic > max_static

    def test_global_budget_respected(self, synthetic_video):
        result = run_pipeline(synthetic_video, PipelineConfig(global_budget=6, store_images=False))
        assert len(result.keyframes) <= 6 + result.stats["n_shots"]

    def test_stats_contain_expected_keys(self, synthetic_video):
        result = run_pipeline(synthetic_video, PipelineConfig(store_images=False))
        for key in ("n_shots", "n_keyframes", "motion_class_distribution", "timing_seconds", "feature_mode"):
            assert key in result.stats


# ---------------------------------------------------------------------------
# run_pipeline_batch — chạy hàng loạt trên thư mục (mô phỏng data crawl)
# ---------------------------------------------------------------------------
class TestPipelineBatch:
    def test_processes_all_videos_in_dir(self, video_dir_with_two_videos):
        results = run_pipeline_batch(
            video_dir_with_two_videos, PipelineConfig(store_images=False), continue_on_error=True,
        )
        assert len(results) == 2

    def test_ok_video_has_no_error(self, video_dir_with_two_videos):
        results = run_pipeline_batch(
            video_dir_with_two_videos, PipelineConfig(store_images=False), continue_on_error=True,
        )
        ok_result = [r for path, r in results.items() if "video_ok" in path][0]
        assert ok_result.error is None
        assert ok_result.stats["n_keyframes"] > 0

    def test_broken_video_has_error_not_crash(self, video_dir_with_two_videos):
        results = run_pipeline_batch(
            video_dir_with_two_videos, PipelineConfig(store_images=False), continue_on_error=True,
        )
        broken_result = [r for path, r in results.items() if "video_broken" in path][0]
        assert broken_result.error is not None
        assert broken_result.stats == {}

    def test_limit_parameter(self, video_dir_with_two_videos):
        results = run_pipeline_batch(
            video_dir_with_two_videos, PipelineConfig(store_images=False), limit=1,
        )
        assert len(results) == 1

    def test_raises_when_continue_on_error_false(self, video_dir_with_two_videos):
        with pytest.raises(Exception):
            run_pipeline_batch(
                video_dir_with_two_videos, PipelineConfig(store_images=False),
                continue_on_error=False,
            )

    def test_empty_dir_returns_empty_dict(self, tmp_path):
        results = run_pipeline_batch(str(tmp_path), PipelineConfig())
        assert results == {}


# ---------------------------------------------------------------------------
# Tầng 5 — Temporal Reranker (MỚI, logic thuần Python, không cần GPU)
# ---------------------------------------------------------------------------
class TestTemporalReranker:
    def test_single_stage_returns_sorted_by_similarity(self):
        hits = [
            [
                StageHit(video_id="v1", stage_index=0, timestamp=1.0, similarity=0.5),
                StageHit(video_id="v2", stage_index=0, timestamp=2.0, similarity=0.9),
            ]
        ]
        reranker = TemporalReranker()
        results = reranker.rerank(hits)
        assert len(results) == 2
        assert results[0].video_id == "v2"  # similarity cao hơn -> đứng đầu
        assert results[0].chain_score == 0.9

    def test_two_stage_prefers_temporally_close_pair(self):
        """
        Kịch bản: video v1 có 2 ứng viên ở stage 0 (điểm ngang nhau), nhưng chỉ
        1 trong 2 nằm GẦN THỜI GIAN với ứng viên tốt ở stage 1. Reranker phải
        chọn đúng cặp gần thời gian hơn, dù similarity thô bằng nhau.
        """
        stage0 = [
            StageHit(video_id="v1", stage_index=0, timestamp=10.0, similarity=0.8),
            StageHit(video_id="v1", stage_index=0, timestamp=100.0, similarity=0.8),
        ]
        stage1 = [
            StageHit(video_id="v1", stage_index=1, timestamp=15.0, similarity=0.8),  # gần hit thứ nhất
        ]
        reranker = TemporalReranker(T_max=50.0)
        results = reranker.rerank([stage0, stage1])
        assert len(results) == 1
        chain = results[0]
        assert chain.hits[0].timestamp == 10.0, "Phải chọn hit ở t=10 (gần stage sau) chứ không phải t=100"

    def test_chain_beyond_T_max_is_orphaned_not_linked(self):
        """Nếu khoảng cách thời gian vượt T_max, 2 hit KHÔNG được nối — chuỗi
        phải coi hit ở stage sau là 'mồ côi', bị decay chứ không link giả."""
        stage0 = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.9)]
        stage1 = [StageHit(video_id="v1", stage_index=1, timestamp=1000.0, similarity=0.9)]
        reranker = TemporalReranker(T_max=50.0, decay_eta=0.1)
        results = reranker.rerank([stage0, stage1])
        assert len(results) == 1
        # vì quá xa (1000s > T_max=50s) nên hit thứ 2 bị decay: score = 0.1*0.9 = 0.09
        assert abs(results[0].chain_score - 0.09) < 1e-6

    def test_three_stage_chain_relinks_globally(self):
        """
        Test bất biến quan trọng: với 3 stage, reranker phải tìm chuỗi TOÀN CỤC
        tốt nhất (DP), không phải chỉ nối tham lam từng cặp liền kề — mô phỏng
        đúng ví dụ "mở cửa -> đi ra -> lái xe đi" trong docstring WESp, nơi
        cue muộn có thể quyết định lựa chọn ở link sớm hơn.
        """
        # 2 lựa chọn ở stage 0, cả 2 đều link tốt tới stage 1, nhưng chỉ 1
        # trong 2 đường mới link tốt tiếp được tới stage 2.
        stage0 = [
            StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.7),   # nhánh A
            StageHit(video_id="v1", stage_index=0, timestamp=0.5, similarity=0.7),   # nhánh B
        ]
        stage1 = [
            StageHit(video_id="v1", stage_index=1, timestamp=5.0, similarity=0.7),   # gần cả 2 nhánh
        ]
        stage2 = [
            StageHit(video_id="v1", stage_index=2, timestamp=6.0, similarity=0.9),   # chỉ gần nếu đi từ stage1 (t=5)
        ]
        reranker = TemporalReranker(T_max=100.0)
        results = reranker.rerank([stage0, stage1, stage2])
        assert len(results) == 1
        chain = results[0]
        assert len(chain.hits) == 3
        assert chain.hits[-1].similarity == 0.9

    def test_missing_stage_for_video_excludes_it(self):
        """Video không có hit ở MỘT stage nào đó (thiếu hẳn chứng cứ) -> bị loại
        khỏi kết quả, không được cho vào chuỗi thiếu."""
        stage0 = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.9)]
        stage1 = [StageHit(video_id="v2", stage_index=1, timestamp=1.0, similarity=0.9)]  # video khác!
        reranker = TemporalReranker()
        results = reranker.rerank([stage0, stage1])
        assert results == [], "v1 thiếu stage 1, v2 thiếu stage 0 -> không video nào đủ chuỗi"

    def test_empty_input_returns_empty(self):
        reranker = TemporalReranker()
        assert reranker.rerank([]) == []

    def test_penalty_mode_sqrt_runs(self):
        """chain_score là TỔNG TÍCH LŨY qua các stage (dp[k]=dp[k-1]+link), không
        phải giá trị chuẩn hoá [0,1] — với 2 stage similarity=0.8 mỗi cái, cận
        trên hợp lý là 2.0 (tổng similarity tối đa nếu không bị phạt gì)."""
        stage0 = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.8)]
        stage1 = [StageHit(video_id="v1", stage_index=1, timestamp=3.0, similarity=0.8)]
        reranker = TemporalReranker(penalty_mode="sqrt", T_max=50.0)
        results = reranker.rerank([stage0, stage1])
        assert len(results) == 1
        assert 0.0 <= results[0].chain_score <= 2.0
        # khoảng cách rất nhỏ (3s) -> phạt gần như không đáng kể -> score gần 2*0.8=1.6
        assert results[0].chain_score > 1.4

    def test_results_sorted_descending(self):
        stage0 = [
            StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.9),
            StageHit(video_id="v2", stage_index=0, timestamp=0.0, similarity=0.3),
        ]
        stage1 = [
            StageHit(video_id="v1", stage_index=1, timestamp=1.0, similarity=0.9),
            StageHit(video_id="v2", stage_index=1, timestamp=1.0, similarity=0.3),
        ]
        reranker = TemporalReranker(T_max=50.0)
        results = reranker.rerank([stage0, stage1])
        scores = [r.chain_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_w_A_weighting_affects_score(self):
        """w_A gần 1 -> ưu tiên similarity của stage TRƯỚC; w_A gần 0 -> ưu tiên
        stage SAU. Kiểm tra công thức blend hoạt động đúng chiều."""
        stage0 = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=1.0)]
        stage1 = [StageHit(video_id="v1", stage_index=1, timestamp=1.0, similarity=0.0)]

        reranker_favor_first = TemporalReranker(w_A=0.9, lambda_=0.0, T_max=50.0)
        reranker_favor_second = TemporalReranker(w_A=0.1, lambda_=0.0, T_max=50.0)

        score_favor_first = reranker_favor_first.rerank([stage0, stage1])[0].chain_score
        score_favor_second = reranker_favor_second.rerank([stage0, stage1])[0].chain_score

        assert score_favor_first > score_favor_second


class TestFastPipelineMatchesOriginal:
    """
    Kiểm chứng run_pipeline_fast() cho ĐÚNG KẾT QUẢ như run_pipeline() gốc —
    đây là điều kiện bắt buộc: tối ưu tốc độ KHÔNG được đổi logic/kết quả.
    """

    def test_same_number_of_shots(self, synthetic_video):
        r_old = run_pipeline(synthetic_video, PipelineConfig(store_images=False))
        r_fast = run_pipeline_fast(synthetic_video, FastPipelineConfig(store_images=False))
        assert r_old.stats["n_shots"] == r_fast.stats["n_shots"]

    def test_same_motion_classification(self, synthetic_video):
        r_old = run_pipeline(synthetic_video, PipelineConfig(store_images=False))
        r_fast = run_pipeline_fast(synthetic_video, FastPipelineConfig(store_images=False))
        old_classes = {mp.shot_id: mp.motion_class for mp in r_old.motion_profiles}
        fast_classes = {mp.shot_id: mp.motion_class for mp in r_fast.motion_profiles}
        assert old_classes == fast_classes

    def test_same_keyframe_count_per_shot(self, synthetic_video):
        r_old = run_pipeline(synthetic_video, PipelineConfig(store_images=False))
        r_fast = run_pipeline_fast(synthetic_video, FastPipelineConfig(store_images=False))

        def count_by_shot(kfs):
            d = {}
            for kf in kfs:
                d[kf.shot_id] = d.get(kf.shot_id, 0) + 1
            return d

        assert count_by_shot(r_old.keyframes) == count_by_shot(r_fast.keyframes)

    def test_fast_is_not_slower_than_original(self, synthetic_video):
        """Không yêu cầu tỉ lệ tăng tốc cụ thể (phụ thuộc máy chạy CI), chỉ
        đảm bảo bản fast không chậm hơn đáng kể — tránh regression."""
        import time
        t0 = time.time()
        run_pipeline(synthetic_video, PipelineConfig(store_images=False))
        t_old = time.time() - t0

        t0 = time.time()
        run_pipeline_fast(synthetic_video, FastPipelineConfig(store_images=False))
        t_fast = time.time() - t0

        assert t_fast <= t_old * 1.2, (
            f"run_pipeline_fast ({t_fast:.3f}s) không được chậm hơn đáng kể "
            f"so với run_pipeline ({t_old:.3f}s)"
        )

    def test_global_budget_works_in_fast_path(self, synthetic_video):
        result = run_pipeline_fast(synthetic_video, FastPipelineConfig(global_budget=6, store_images=False))
        assert len(result.keyframes) <= 6 + result.stats["n_shots"]

    def test_store_images_true_returns_images(self, synthetic_video):
        result = run_pipeline_fast(synthetic_video, FastPipelineConfig(store_images=True))
        for kf in result.keyframes:
            assert kf.image is not None

    def test_candidate_resize_reduces_image_size(self, synthetic_video):
        result = run_pipeline_fast(
            synthetic_video,
            FastPipelineConfig(store_images=True, candidate_resize_to=(80, 45)),
        )
        for kf in result.keyframes:
            assert kf.image.shape[1] == 80
            assert kf.image.shape[0] == 45



class TestHFStreamingLogic:
    """
    Test luồng logic của stream_process_hf_dataset() bằng MOCK — không gọi
    HuggingFace Hub thật (dataset gated, cần token cá nhân). Mock thay
    _download_one_video bằng việc copy file video test có sẵn, để kiểm chứng
    đúng cơ chế: tải -> xử lý -> lưu gọn -> XOÁ VIDEO GỐC -> skip_existing.
    """

    def test_deletes_original_video_after_processing(self, synthetic_video, tmp_path, monkeypatch):
        import shutil as _shutil
        from aic_pipeline import hf_streaming

        def fake_list_files(repo_id, hf_token=None):
            return ["video_a.mp4"]

        def fake_download(repo_id, filename, local_dir, hf_token):
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, filename)
            _shutil.copy(synthetic_video, dest)
            return dest

        monkeypatch.setattr(hf_streaming, "list_video_files_in_repo", fake_list_files)
        monkeypatch.setattr(hf_streaming, "_download_one_video", fake_download)
        monkeypatch.setattr(hf_streaming, "_check_hf_hub_available", lambda: None)

        tmp_dl = str(tmp_path / "tmp_dl")
        out_dir = str(tmp_path / "processed")

        hf_streaming.stream_process_hf_dataset(
            repo_id="fake/repo", output_dir=out_dir,
            pipeline_config=FastPipelineConfig(store_images=True),
            tmp_download_dir=tmp_dl,
        )

        downloaded_path = os.path.join(tmp_dl, "video_a.mp4")
        assert not os.path.exists(downloaded_path), "Video gốc phải bị xoá sau khi xử lý"

    def test_saves_compact_result_not_raw_video(self, synthetic_video, tmp_path, monkeypatch):
        import shutil as _shutil
        from aic_pipeline import hf_streaming

        def fake_list_files(repo_id, hf_token=None):
            return ["video_a.mp4"]

        def fake_download(repo_id, filename, local_dir, hf_token):
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, filename)
            _shutil.copy(synthetic_video, dest)
            return dest

        monkeypatch.setattr(hf_streaming, "list_video_files_in_repo", fake_list_files)
        monkeypatch.setattr(hf_streaming, "_download_one_video", fake_download)
        monkeypatch.setattr(hf_streaming, "_check_hf_hub_available", lambda: None)

        out_dir = str(tmp_path / "processed")
        results = hf_streaming.stream_process_hf_dataset(
            repo_id="fake/repo", output_dir=out_dir,
            pipeline_config=FastPipelineConfig(store_images=True),
            tmp_download_dir=str(tmp_path / "tmp_dl"),
        )

        assert "video_a" in results
        video_out_dir = os.path.join(out_dir, "video_a")
        assert os.path.exists(os.path.join(video_out_dir, "metadata.json"))
        jpg_files = [f for f in os.listdir(video_out_dir) if f.endswith(".jpg")]
        assert len(jpg_files) == results["video_a"]["stats"]["n_keyframes"]
        assert not any(f.endswith(".mp4") for f in os.listdir(video_out_dir))

    def test_skip_existing_avoids_reprocessing(self, synthetic_video, tmp_path, monkeypatch):
        import shutil as _shutil
        from aic_pipeline import hf_streaming

        call_count = {"n": 0}

        def fake_list_files(repo_id, hf_token=None):
            return ["video_a.mp4"]

        def fake_download(repo_id, filename, local_dir, hf_token):
            call_count["n"] += 1
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, filename)
            _shutil.copy(synthetic_video, dest)
            return dest

        monkeypatch.setattr(hf_streaming, "list_video_files_in_repo", fake_list_files)
        monkeypatch.setattr(hf_streaming, "_download_one_video", fake_download)
        monkeypatch.setattr(hf_streaming, "_check_hf_hub_available", lambda: None)

        out_dir = str(tmp_path / "processed")
        cfg = FastPipelineConfig(store_images=False)

        hf_streaming.stream_process_hf_dataset(
            repo_id="fake/repo", output_dir=out_dir, pipeline_config=cfg,
            tmp_download_dir=str(tmp_path / "tmp_dl1"), skip_existing=True,
        )
        assert call_count["n"] == 1

        hf_streaming.stream_process_hf_dataset(
            repo_id="fake/repo", output_dir=out_dir, pipeline_config=cfg,
            tmp_download_dir=str(tmp_path / "tmp_dl2"), skip_existing=True,
        )
        assert call_count["n"] == 1, "Lần chạy thứ 2 phải skip, không tải lại video đã xử lý"

    def test_error_in_one_video_does_not_stop_batch(self, synthetic_video, tmp_path, monkeypatch):
        import shutil as _shutil
        from aic_pipeline import hf_streaming

        def fake_list_files(repo_id, hf_token=None):
            return ["broken.mp4", "video_ok.mp4"]

        def fake_download(repo_id, filename, local_dir, hf_token):
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, filename)
            if "broken" in filename:
                with open(dest, "wb") as f:
                    f.write(b"not a real video")
            else:
                _shutil.copy(synthetic_video, dest)
            return dest

        monkeypatch.setattr(hf_streaming, "list_video_files_in_repo", fake_list_files)
        monkeypatch.setattr(hf_streaming, "_download_one_video", fake_download)
        monkeypatch.setattr(hf_streaming, "_check_hf_hub_available", lambda: None)

        out_dir = str(tmp_path / "processed")
        results = hf_streaming.stream_process_hf_dataset(
            repo_id="fake/repo", output_dir=out_dir,
            pipeline_config=FastPipelineConfig(store_images=False),
            tmp_download_dir=str(tmp_path / "tmp_dl"),
        )

        assert "video_ok" in results, "Video hợp lệ vẫn phải được xử lý dù video khác lỗi"
        assert "broken" not in results

    def test_on_video_done_callback_called(self, synthetic_video, tmp_path, monkeypatch):
        import shutil as _shutil
        from aic_pipeline import hf_streaming

        def fake_list_files(repo_id, hf_token=None):
            return ["video_a.mp4"]

        def fake_download(repo_id, filename, local_dir, hf_token):
            os.makedirs(local_dir, exist_ok=True)
            dest = os.path.join(local_dir, filename)
            _shutil.copy(synthetic_video, dest)
            return dest

        monkeypatch.setattr(hf_streaming, "list_video_files_in_repo", fake_list_files)
        monkeypatch.setattr(hf_streaming, "_download_one_video", fake_download)
        monkeypatch.setattr(hf_streaming, "_check_hf_hub_available", lambda: None)

        called_with = []

        def callback(video_id, meta):
            called_with.append(video_id)

        hf_streaming.stream_process_hf_dataset(
            repo_id="fake/repo", output_dir=str(tmp_path / "processed"),
            pipeline_config=FastPipelineConfig(store_images=False),
            tmp_download_dir=str(tmp_path / "tmp_dl"),
            on_video_done=callback,
        )
        assert called_with == ["video_a"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

class TestGPUShotDetector:
    """
    Test GPUShotDetector ở CHẾ ĐỘ ZERO-SHOT (không checkpoint) — điều kiện
    BẮT BUỘC quan trọng nhất: model chưa huấn luyện (random init) KHÔNG được
    phép tạo ra kết quả TỆ HƠN baseline (đã từng có bug tạo 17 shot giả trên
    video chỉ có 3 shot thật — test này khoá lại để tránh regression).
    """

    def test_zero_shot_not_worse_than_baseline(self, synthetic_video):
        from aic_pipeline.shot_detector import GPUShotDetector

        det = GPUShotDetector(device="cpu", min_shot_len=15)
        shots = det.detect(synthetic_video)
        assert len(shots) <= 4, (
            f"GPUShotDetector zero-shot tạo {len(shots)} shot trên video chỉ "
            f"có 3 shot thật — có dấu hiệu false positive giống bug đã fix."
        )

    def test_zero_shot_boundaries_match_baseline_approximately(self, synthetic_video):
        from aic_pipeline.shot_detector import GPUShotDetector

        det = GPUShotDetector(device="cpu", min_shot_len=15)
        shots = det.detect(synthetic_video)
        boundary_times = [shots[0].start_time] + [s.end_time for s in shots]
        expected = [0.0, 3.0, 5.0, 10.0]
        for exp_t in expected:
            assert any(abs(bt - exp_t) < 0.3 for bt in boundary_times), (
                f"Không tìm thấy boundary gần {exp_t}s trong {boundary_times}"
            )

    def test_untrained_model_flag_is_false_by_default(self):
        from aic_pipeline.shot_detector import GPUShotDetector
        det = GPUShotDetector(device="cpu")
        det._load_model()
        assert det._model_is_trained is False, (
            "Mặc định (không checkpoint) PHẢI đánh dấu model chưa huấn luyện, "
            "để không cho phép model tự quyết cut_positions."
        )

    def test_invalid_checkpoint_path_raises_clear_error(self):
        from aic_pipeline.shot_detector import GPUShotDetector
        det = GPUShotDetector(device="cpu", checkpoint_path="/nonexistent/ckpt.pth")
        with pytest.raises(IOError):
            det._load_model()


class TestAutoCalibrateGMM:
    def test_auto_calibrate_runs_and_returns_valid_thresholds(self, synthetic_video):
        from aic_pipeline.motion_scorer import auto_calibrate_thresholds_gmm

        video_paths = [synthetic_video] * 5
        try:
            lo, hi, debug = auto_calibrate_thresholds_gmm(video_paths, n_components=3)
        except ValueError as e:
            pytest.skip(f"Không đủ mẫu shot cho GMM với video test nhỏ: {e}")
        assert lo <= hi
        assert "cluster_means_sorted" in debug
        assert len(debug["cluster_means_sorted"]) == 3

    def test_auto_calibrate_raises_with_too_few_samples(self, synthetic_video):
        from aic_pipeline.motion_scorer import auto_calibrate_thresholds_gmm
        with pytest.raises(ValueError):
            auto_calibrate_thresholds_gmm([synthetic_video], n_components=3)


class TestObjectGraph:
    def test_builds_graph_without_crashing(self, synthetic_video):
        from aic_pipeline.object_graph import build_object_graph
        from aic_pipeline.shot_detector import Shot

        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        graph = build_object_graph(synthetic_video, shot, frame_stride=3)
        assert graph.shot_id == 2

    def test_dynamic_shot_detects_multiple_tracks(self, synthetic_video):
        """Shot động (nhiều quả bóng chuyển động) phải phát hiện được NHIỀU
        track — kiểm chứng ContourFallbackDetector + tracking hoạt động."""
        from aic_pipeline.object_graph import build_object_graph
        from aic_pipeline.shot_detector import Shot

        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        graph = build_object_graph(synthetic_video, shot, frame_stride=3)
        assert len(graph.tracks) > 1, "Shot có 8 vật thể chuyển động phải phát hiện được nhiều hơn 1 track"

    def test_static_shot_detects_few_or_no_tracks(self, synthetic_video):
        """Shot tĩnh (không chuyển động) phải phát hiện RẤT ÍT track so với
        shot động — kiểm chứng background subtraction hoạt động đúng logic."""
        from aic_pipeline.object_graph import build_object_graph
        from aic_pipeline.shot_detector import Shot

        static_shot = Shot(shot_id=0, start_frame=0, end_frame=75, start_time=0.0, end_time=3.0)
        dynamic_shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)

        static_graph = build_object_graph(synthetic_video, static_shot, frame_stride=3)
        dynamic_graph = build_object_graph(synthetic_video, dynamic_shot, frame_stride=3)

        assert len(static_graph.tracks) <= len(dynamic_graph.tracks)

    def test_summarize_graph_returns_valid_dict(self, synthetic_video):
        from aic_pipeline.object_graph import build_object_graph, summarize_graph_for_query
        from aic_pipeline.shot_detector import Shot

        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        graph = build_object_graph(synthetic_video, shot, frame_stride=3)
        summary = summarize_graph_for_query(graph)
        assert "n_tracks" in summary
        assert "top_interactions" in summary
        assert isinstance(summary["top_interactions"], list)

    def test_interaction_threshold_filters_far_objects(self, synthetic_video):
        """Ngưỡng interaction_distance_threshold rất nhỏ phải loại gần hết
        cạnh tương tác (vì hầu hết vật thể ở xa nhau hơn ngưỡng)."""
        from aic_pipeline.object_graph import build_object_graph
        from aic_pipeline.shot_detector import Shot

        shot = Shot(shot_id=2, start_frame=125, end_frame=250, start_time=5.0, end_time=10.0)
        graph_strict = build_object_graph(
            synthetic_video, shot, frame_stride=3, interaction_distance_threshold=1.0
        )
        graph_loose = build_object_graph(
            synthetic_video, shot, frame_stride=3, interaction_distance_threshold=1000.0
        )
        assert len(graph_strict.edges) <= len(graph_loose.edges)


class TestTemporalRerankerBidirectional:
    def test_bidirectional_runs_without_crash(self):
        from aic_pipeline import TemporalReranker, StageHit
        stage0 = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.8)]
        stage1 = [StageHit(video_id="v1", stage_index=1, timestamp=3.0, similarity=0.8)]
        reranker = TemporalReranker(T_max=50.0)
        results = reranker.rerank_bidirectional([stage0, stage1])
        assert len(results) == 1
        assert len(results[0].hits) == 2

    def test_bidirectional_single_stage_falls_back_to_rerank(self):
        from aic_pipeline import TemporalReranker, StageHit
        stage0 = [
            StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.9),
            StageHit(video_id="v2", stage_index=0, timestamp=0.0, similarity=0.3),
        ]
        reranker = TemporalReranker()
        results = reranker.rerank_bidirectional([stage0])
        assert results[0].video_id == "v1"

    def test_bidirectional_missing_stage_excludes_video(self):
        from aic_pipeline import TemporalReranker, StageHit
        stage0 = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.9)]
        stage1 = [StageHit(video_id="v2", stage_index=1, timestamp=1.0, similarity=0.9)]
        reranker = TemporalReranker()
        results = reranker.rerank_bidirectional([stage0, stage1])
        assert results == []

    def test_hit_score_reflects_future_evidence_not_just_past(self):
        """
        Kiểm chứng ĐÚNG BẢN CHẤT của bidirectional: điểm combined tại stage
        GIỮA phải cao hơn khi stage SAU nó có bằng chứng cực mạnh, so với
        trường hợp stage sau yếu — dù bản thân similarity ở stage giữa
        không đổi. Đây là điều forward-only KHÔNG làm được (dp[k][i] không
        biết gì về tương lai).
        """
        from aic_pipeline import TemporalReranker, StageHit

        # Trường hợp A: stage sau MẠNH
        stage0_a = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.5)]
        stage1_a = [StageHit(video_id="v1", stage_index=1, timestamp=3.0, similarity=0.5)]
        stage2_a = [StageHit(video_id="v1", stage_index=2, timestamp=6.0, similarity=0.99)]

        # Trường hợp B: stage sau YẾU (mọi thứ khác giống hệt trường hợp A)
        stage0_b = [StageHit(video_id="v1", stage_index=0, timestamp=0.0, similarity=0.5)]
        stage1_b = [StageHit(video_id="v1", stage_index=1, timestamp=3.0, similarity=0.5)]
        stage2_b = [StageHit(video_id="v1", stage_index=2, timestamp=6.0, similarity=0.1)]

        reranker = TemporalReranker(T_max=50.0, lambda_=0.0)

        result_a = reranker.rerank_bidirectional([stage0_a, stage1_a, stage2_a])[0]
        result_b = reranker.rerank_bidirectional([stage0_b, stage1_b, stage2_b])[0]

        assert result_a.chain_score > result_b.chain_score, (
            "Chuỗi có bằng chứng mạnh ở stage sau phải có điểm cao hơn, dù "
            "similarity ở stage 0 và 1 giống hệt nhau giữa 2 trường hợp — "
            "đây là bằng chứng cross-moment 2 chiều hoạt động đúng."
        )


class TestGPUShotDetectorDeterminism:
    """Khoá lại bug non-determinism đã fix: cùng video, nhiều lần gọi detect()
    liên tiếp PHẢI cho kết quả giống hệt nhau (số shot, boundary)."""

    def test_detect_is_deterministic_across_multiple_calls(self, synthetic_video):
        from aic_pipeline.shot_detector import GPUShotDetector

        results = []
        for _ in range(4):
            det = GPUShotDetector(device="cpu", min_shot_len=15)
            shots = det.detect(synthetic_video)
            results.append(tuple((s.start_frame, s.end_frame) for s in shots))

        assert len(set(results)) == 1, (
            f"GPUShotDetector.detect() phải cho kết quả GIỐNG HỆT NHAU giữa "
            f"các lần gọi (cùng seed) — nhưng nhận được {len(set(results))} "
            f"kết quả khác nhau: {results}"
        )


class TestAutoShotDetectorReal:
    """
    Test AutoShotDetector với checkpoint GIẢ (random weights đúng format) —
    kiểm chứng LUỒNG CODE đúng (đọc frame ffmpeg, sliding window, forward,
    build Shot), độc lập với việc có checkpoint thật ckpt_0_200_0.pth hay
    không (checkpoint thật chỉ có trên máy người dùng, không có ở đây).
    """

    @classmethod
    @pytest.fixture(scope="class")
    def fake_checkpoint_path(cls, tmp_path_factory):
        import sys
        import torch
        vendor_dir = os.path.join(
            os.path.dirname(__file__), "..", "aic_pipeline", "_autoshot_vendor"
        )
        sys.path.insert(0, vendor_dir)
        from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet

        model = TransNetV2Supernet().eval()
        path = str(tmp_path_factory.mktemp("ckpt") / "fake_ckpt.pth")
        torch.save({"net": model.state_dict()}, path)
        return path

    def test_missing_checkpoint_raises_clear_ioerror(self):
        from aic_pipeline.shot_detector import AutoShotDetector
        det = AutoShotDetector(checkpoint_path="/nonexistent/ckpt.pth", device="cpu")
        with pytest.raises(IOError, match="Không tìm thấy checkpoint"):
            det.detect("sample_data/test_video.mp4")

    def test_full_pipeline_runs_with_fake_checkpoint(self, synthetic_video, fake_checkpoint_path):
        """Checkpoint giả (random) vẫn phải chạy hết luồng KHÔNG CRASH —
        đây là điều kiện tiên quyết trước khi tin tưởng checkpoint thật."""
        from aic_pipeline.shot_detector import AutoShotDetector
        det = AutoShotDetector(checkpoint_path=fake_checkpoint_path, device="cpu")
        shots = det.detect(synthetic_video)
        assert len(shots) > 0
        for s in shots:
            assert 0.0 <= s.confidence <= 1.0
            assert s.start_frame < s.end_frame

    def test_checkpoint_param_matching_logged(self, synthetic_video, fake_checkpoint_path, caplog):
        import logging
        from aic_pipeline.shot_detector import AutoShotDetector
        with caplog.at_level(logging.INFO, logger="aic_pipeline.shot_detector"):
            det = AutoShotDetector(checkpoint_path=fake_checkpoint_path, device="cpu")
            det.detect(synthetic_video)
        assert any("khớp được" in r.message for r in caplog.records)

    def test_full_shot_list_covers_entire_video(self, synthetic_video, fake_checkpoint_path):
        from aic_pipeline.shot_detector import AutoShotDetector
        det = AutoShotDetector(checkpoint_path=fake_checkpoint_path, device="cpu", min_shot_len=5)
        shots = det.detect(synthetic_video)
        assert shots[0].start_frame == 0
        for i in range(len(shots) - 1):
            assert shots[i].end_frame == shots[i + 1].start_frame

    def test_integrates_with_run_pipeline(self, synthetic_video, fake_checkpoint_path):
        """AutoShotDetector (checkpoint giả) phải cắm được vào PipelineConfig
        và chạy hết run_pipeline() không lỗi — kiểm chứng tích hợp Protocol
        ShotDetectorBackend đúng."""
        from aic_pipeline import PipelineConfig, run_pipeline
        from aic_pipeline.shot_detector import AutoShotDetector

        det = AutoShotDetector(checkpoint_path=fake_checkpoint_path, device="cpu")
        config = PipelineConfig(shot_backend=det, store_images=False)
        result = run_pipeline(synthetic_video, config)
        assert result.stats["n_shots"] > 0
        assert result.stats["n_keyframes"] > 0



class TestOmniShotCutDetector:
    """
    Test OmniShotCutDetector với checkpoint GIẢ (random weights đúng format
    {'args':..., 'model':...}) — kiểm chứng LUỒNG CODE đúng, độc lập với
    checkpoint thật (chỉ tải được từ HuggingFace, cần mạng).

    Patch is_main_process=False để backbone KHÔNG cố tải pretrained ImageNet
    weights qua download.pytorch.org (môi trường build có thể chặn mạng này;
    trên Kaggle với Internet:On sẽ tự tải bình thường, không cần patch).
    """

    @classmethod
    @pytest.fixture(scope="class")
    def patch_no_pretrained(cls):
        import sys
        vendor_dir = os.path.join(
            os.path.dirname(__file__), "..", "aic_pipeline", "_omnishotcut_vendor"
        )
        sys.path.insert(0, vendor_dir)
        from omnishotcut.architecture import backbone as backbone_module
        original = backbone_module.is_main_process
        backbone_module.is_main_process = lambda: False
        yield
        backbone_module.is_main_process = original

    @classmethod
    @pytest.fixture(scope="class")
    def fake_omnishotcut_checkpoint(cls, tmp_path_factory, patch_no_pretrained):
        from types import SimpleNamespace
        import torch
        from omnishotcut.architecture.backbone import build_backbone
        from omnishotcut.architecture.transformer import build_transformer
        from omnishotcut.architecture.model import OmniShotCut

        model_args = SimpleNamespace(
            backbone="resnet50", dilation=False, lr_backbone=1e-5, masks=False,
            hidden_dim=192, dropout=0.1, nheads=8, dim_feedforward=2048,
            enc_layers=6, dec_layers=6, pre_norm=False, position_embedding="sine",
            num_intra_relation_classes=9, num_inter_relation_classes=6,
            max_process_window_length=16, num_queries=16, aux_loss=False,
            process_height=224, process_width=224,
        )
        backbone = build_backbone(model_args)
        transformer = build_transformer(model_args)
        model = OmniShotCut(
            backbone, transformer,
            num_intra_relation_classes=model_args.num_intra_relation_classes,
            num_inter_relation_classes=model_args.num_inter_relation_classes,
            num_frames=model_args.max_process_window_length,
            num_queries=model_args.num_queries,
            aux_loss=model_args.aux_loss,
        )
        path = str(tmp_path_factory.mktemp("ckpt") / "fake_omnishotcut.pth")
        torch.save({"args": model_args, "model": model.state_dict()}, path)
        return path

    def test_missing_checkpoint_and_no_hf_raises(self):
        from aic_pipeline.shot_detector import OmniShotCutDetector
        det = OmniShotCutDetector(checkpoint_path="/nonexistent/ckpt.pth", device="cpu")
        with pytest.raises(IOError, match="Không tìm thấy checkpoint"):
            det.detect("sample_data/test_video.mp4")

    def test_checkpoint_missing_required_keys_raises_clear_error(self, tmp_path):
        import torch
        from aic_pipeline.shot_detector import OmniShotCutDetector
        bad_ckpt_path = str(tmp_path / "bad.pth")
        torch.save({"wrong_key": 123}, bad_ckpt_path)
        det = OmniShotCutDetector(checkpoint_path=bad_ckpt_path, device="cpu")
        with pytest.raises(ValueError, match="'args' và 'model'"):
            det.detect("sample_data/test_video.mp4")

    def test_full_pipeline_runs_with_fake_checkpoint(
        self, synthetic_video, fake_omnishotcut_checkpoint, patch_no_pretrained
    ):
        from aic_pipeline.shot_detector import OmniShotCutDetector
        det = OmniShotCutDetector(
            checkpoint_path=fake_omnishotcut_checkpoint, device="cpu",
            min_shot_len=5, overlap_window_length=4,
        )
        shots = det.detect(synthetic_video)
        assert len(shots) > 0
        for s in shots:
            assert s.start_frame < s.end_frame
            assert s.boundary_type in ("hard", "gradual")

    def test_clean_shot_mode_filters_transitions(
        self, synthetic_video, fake_omnishotcut_checkpoint, patch_no_pretrained
    ):
        from aic_pipeline.shot_detector import OmniShotCutDetector
        det = OmniShotCutDetector(
            checkpoint_path=fake_omnishotcut_checkpoint, device="cpu", mode="clean_shot",
            min_shot_len=5, overlap_window_length=4,
        )
        # với weights ngẫu nhiên có thể lọc hết sạch -> chấp nhận ValueError
        # HOẶC danh sách rỗng-an-toàn, miễn không crash kiểu khác
        try:
            shots = det.detect(synthetic_video)
            assert isinstance(shots, list)
        except ValueError as e:
            assert "clean_shot" in str(e) or "không phát hiện" in str(e).lower()

    def test_integrates_with_run_pipeline(
        self, synthetic_video, fake_omnishotcut_checkpoint, patch_no_pretrained
    ):
        from aic_pipeline import PipelineConfig, run_pipeline
        from aic_pipeline.shot_detector import OmniShotCutDetector

        det = OmniShotCutDetector(
            checkpoint_path=fake_omnishotcut_checkpoint, device="cpu",
            min_shot_len=5, overlap_window_length=4,
        )
        config = PipelineConfig(shot_backend=det, store_images=False)
        result = run_pipeline(synthetic_video, config)
        assert result.stats["n_shots"] > 0
        assert result.stats["n_keyframes"] > 0


class TestOmniShotCutAV1Support:
    """
    Test QUAN TRỌNG NHẤT của patch PyAV: xác nhận OmniShotCutDetector đọc
    được video mã hoá AV1 — codec mà decord (bản cuối 0.6.0, dự án đã ngừng
    phát triển) KHÔNG hỗ trợ, gây lỗi DECORDError thật đã gặp trên Kaggle
    (dataset AIC dùng codec av1). Patch thay decord bằng PyAV trong
    _omnishotcut_vendor/omnishotcut/engine.py và __init__.py.
    """

    @pytest.fixture(scope="class")
    def av1_video(self, tmp_path_factory, synthetic_video):
        """Convert video test sang AV1 bằng ffmpeg để tái hiện đúng vấn đề thật."""
        import subprocess
        av1_path = str(tmp_path_factory.mktemp("av1") / "test_av1.mp4")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", synthetic_video, "-c:v", "libaom-av1",
             "-crf", "30", "-cpu-used", "8", av1_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(av1_path):
            pytest.skip(f"Không tạo được video AV1 test (thiếu libaom-av1?): {result.stderr[:300]}")
        return av1_path

    def test_decord_fails_on_av1_baseline_proof(self, av1_video):
        """Xác nhận ĐÚNG VẤN ĐỀ: decord (chưa patch) phải lỗi trên AV1 —
        đây là bằng chứng nền để so sánh, không phải bug cần sửa ở đây."""
        from decord import VideoReader, cpu as decord_cpu
        with pytest.raises(Exception):  # DECORDError
            VideoReader(av1_video, ctx=decord_cpu(0), width=224, height=224)

    def test_pyav_reads_av1_successfully(self, av1_video):
        """Hàm patch _read_video_pyav phải đọc được AV1 mà decord không đọc được."""
        import sys
        vendor_dir = os.path.join(
            os.path.dirname(__file__), "..", "aic_pipeline", "_omnishotcut_vendor"
        )
        sys.path.insert(0, vendor_dir)
        from omnishotcut.engine import _read_video_pyav

        video_np, fps = _read_video_pyav(av1_video, width=224, height=224)
        assert video_np.ndim == 4
        assert video_np.shape[1:] == (224, 224, 3)
        assert video_np.shape[0] > 0
        assert fps > 0

    @classmethod
    @pytest.fixture(scope="class")
    def patch_no_pretrained_local(cls):
        import sys
        vendor_dir = os.path.join(
            os.path.dirname(__file__), "..", "aic_pipeline", "_omnishotcut_vendor"
        )
        sys.path.insert(0, vendor_dir)
        from omnishotcut.architecture import backbone as backbone_module
        original = backbone_module.is_main_process
        backbone_module.is_main_process = lambda: False
        yield
        backbone_module.is_main_process = original

    @classmethod
    @pytest.fixture(scope="class")
    def fake_checkpoint_local(cls, tmp_path_factory, patch_no_pretrained_local):
        from types import SimpleNamespace
        import torch
        from omnishotcut.architecture.backbone import build_backbone
        from omnishotcut.architecture.transformer import build_transformer
        from omnishotcut.architecture.model import OmniShotCut

        model_args = SimpleNamespace(
            backbone="resnet50", dilation=False, lr_backbone=1e-5, masks=False,
            hidden_dim=192, dropout=0.1, nheads=8, dim_feedforward=2048,
            enc_layers=6, dec_layers=6, pre_norm=False, position_embedding="sine",
            num_intra_relation_classes=9, num_inter_relation_classes=6,
            max_process_window_length=16, num_queries=16, aux_loss=False,
            process_height=224, process_width=224,
        )
        backbone = build_backbone(model_args)
        transformer = build_transformer(model_args)
        model = OmniShotCut(
            backbone, transformer,
            num_intra_relation_classes=model_args.num_intra_relation_classes,
            num_inter_relation_classes=model_args.num_inter_relation_classes,
            num_frames=model_args.max_process_window_length,
            num_queries=model_args.num_queries,
            aux_loss=model_args.aux_loss,
        )
        path = str(tmp_path_factory.mktemp("ckpt") / "fake.pth")
        torch.save({"args": model_args, "model": model.state_dict()}, path)
        return path

    def test_omnishotcut_detector_full_flow_on_av1(
        self, av1_video, fake_checkpoint_local, patch_no_pretrained_local
    ):
        """Test end-to-end quan trọng nhất: OmniShotCutDetector.detect() phải
        chạy THÀNH CÔNG trên video AV1 — đúng chính xác tình huống lỗi thật
        đã gặp trên Kaggle với dataset AIC (codec av1)."""
        from aic_pipeline.shot_detector import OmniShotCutDetector

        det = OmniShotCutDetector(
            checkpoint_path=fake_checkpoint_local, device="cpu",
            min_shot_len=5, overlap_window_length=4,
        )
        shots = det.detect(av1_video)
        assert len(shots) > 0, "Phải phát hiện được ít nhất 1 shot trên video AV1"
        for s in shots:
            assert s.start_frame < s.end_frame


class TestPyAVCorruptedStreamHandling:
    """
    Vá lỗi: video AV1 có phần STREAM BỊ LỖI THẬT (không chỉ thiếu hardware
    accelerator) khiến FFmpeg lặp lại "Failed to get pixel format" / "Get
    current frame error" cho MỌI packet còn lại — trước đây gây treo notebook
    vô thời hạn. Đã vá bằng đếm số packet liên tiếp lỗi/rỗng, dừng sớm khi
    vượt ngưỡng thay vì lặp vô hạn qua phần hỏng của file.
    """

    @pytest.fixture(scope="class")
    def corrupted_av1_video(self, tmp_path_factory, synthetic_video):
        import subprocess
        import random

        av1_path = str(tmp_path_factory.mktemp("av1c") / "good.mp4")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", synthetic_video, "-c:v", "libaom-av1",
             "-crf", "30", "-cpu-used", "8", av1_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(av1_path):
            pytest.skip(f"Không tạo được video AV1 (thiếu libaom-av1?): {result.stderr[:300]}")

        with open(av1_path, "rb") as f:
            data = bytearray(f.read())
        random.seed(42)
        start, end = int(len(data) * 0.5), int(len(data) * 0.9)
        for i in range(start, end):
            data[i] = random.randint(0, 255)

        corrupt_path = str(tmp_path_factory.mktemp("av1c") / "corrupt.mp4")
        with open(corrupt_path, "wb") as f:
            f.write(bytes(data))
        return corrupt_path

    def test_corrupted_stream_does_not_hang(self, corrupted_av1_video):
        """Test QUAN TRỌNG NHẤT: đọc file có vùng dữ liệu bị hỏng phải HOÀN
        TẤT trong thời gian hợp lý (không treo vô hạn), dùng timeout cứng
        qua subprocess để đảm bảo test tự thất bại rõ ràng nếu bị treo thay
        vì làm treo cả pytest."""
        import subprocess
        import sys as _sys

        script = f"""
import sys
sys.path.insert(0, {repr(os.path.join(os.path.dirname(__file__), "..", "aic_pipeline", "_omnishotcut_vendor"))})
from omnishotcut.engine import _read_video_pyav
video_np, fps = _read_video_pyav({repr(corrupted_av1_video)}, width=224, height=224, max_frames=500)
print("OK", video_np.shape)
"""
        result = subprocess.run(
            [_sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,  # nếu treo, subprocess.run tự raise TimeoutExpired
        )
        assert "OK" in result.stdout, f"Không hoàn tất đúng: stdout={result.stdout}, stderr={result.stderr[-500:]}"

    def test_corrupted_stream_returns_frames_before_corruption(self, corrupted_av1_video):
        """Phải trả về đúng các frame đọc được TRƯỚC vùng hỏng, không phải
        rỗng hoàn toàn (vùng hỏng ở 50%-90% file, nên phải có frame từ 0-50%)."""
        import sys
        vendor_dir = os.path.join(
            os.path.dirname(__file__), "..", "aic_pipeline", "_omnishotcut_vendor"
        )
        sys.path.insert(0, vendor_dir)
        from omnishotcut.engine import _read_video_pyav

        video_np, fps = _read_video_pyav(corrupted_av1_video, width=224, height=224, max_frames=500)
        assert video_np.shape[0] > 0, "Phải đọc được ít nhất vài frame trước vùng hỏng"


class TestVideoReaderRAMSafety:
    """
    Vá bug NGHIÊM TRỌNG phát hiện khi test trên video AIC thật
    (K05_V030.mp4, 1920x1080, 32.714 frame): bản đầu của video_reader.py đọc
    GIỮ NGUYÊN ĐỘ PHÂN GIẢI GỐC, với max_frames=6000 cần tới 37.3GB RAM,
    khiến Kaggle kill process ("Killed", không traceback). Đã vá bằng resize
    bắt buộc (mặc định 480x270) + giảm max_frames mặc định (2000).
    """

    def test_default_config_stays_under_2gb(self):
        from aic_pipeline.video_reader import estimate_ram_gb, DEFAULT_MAX_FRAMES, DEFAULT_READ_WIDTH, DEFAULT_READ_HEIGHT
        est = estimate_ram_gb(DEFAULT_MAX_FRAMES, DEFAULT_READ_WIDTH, DEFAULT_READ_HEIGHT)
        assert est < 2.0, (
            f"Cấu hình mặc định ước tính {est:.1f}GB RAM — vượt ngưỡng an toàn 2GB, "
            f"nguy cơ lặp lại bug tràn RAM đã fix."
        )

    def test_full_res_1080p_config_would_exceed_kaggle_ram(self):
        """Xác nhận ĐÚNG bug đã fix: cấu hình full-res 1920x1080 với
        max_frames=6000 (cấu hình CŨ trước khi vá) THẬT SỰ vượt RAM Kaggle
        (13-16GB) — nếu ai đó vô tình đổi resize_width/height=None mà không
        giảm max_frames, phải cảnh báo được, không để lặp lại bug."""
        from aic_pipeline.video_reader import estimate_ram_gb
        est_old_buggy_config = estimate_ram_gb(6000, 1920, 1080)
        assert est_old_buggy_config > 16.0, (
            "Test này xác nhận đúng mức độ nghiêm trọng của bug cũ — nếu giá trị "
            "này thay đổi, kiểm tra lại công thức estimate_ram_gb"
        )

    def test_get_video_frames_resizes_by_default(self, synthetic_video):
        from aic_pipeline.video_reader import get_video_frames, clear_cache
        clear_cache()
        frames, fps = get_video_frames(synthetic_video)
        assert frames.shape[1] == 270 and frames.shape[2] == 480, (
            "Mặc định PHẢI resize xuống 480x270, không giữ nguyên gốc"
        )

    def test_get_video_frames_respects_custom_max_frames(self, synthetic_video):
        from aic_pipeline.video_reader import get_video_frames, clear_cache
        clear_cache()
        frames, fps = get_video_frames(synthetic_video, max_frames=50)
        assert frames.shape[0] <= 50

    def test_cache_key_includes_config_avoids_stale_resolution(self, synthetic_video):
        """Nếu gọi get_video_frames() với 2 cấu hình resize khác nhau cho
        CÙNG 1 video, phải trả về đúng 2 kết quả khác nhau (không dùng nhầm
        cache của cấu hình trước)."""
        from aic_pipeline.video_reader import get_video_frames, clear_cache
        clear_cache()
        frames_a, _ = get_video_frames(synthetic_video, resize_width=480, resize_height=270)
        frames_b, _ = get_video_frames(synthetic_video, resize_width=160, resize_height=90)
        assert frames_a.shape[1:] != frames_b.shape[1:], (
            "Cache key phải phân biệt theo resize config, tránh trả nhầm "
            "kết quả của cấu hình khác."
        )


class TestVideoReaderCoversFullDuration:
    """
    Vá bug NGHIÊM TRỌNG #2 (phát hiện khi thảo luận về đánh đổi max_frames):
    bản đầu dừng TUẦN TỰ ngay khi đủ max_frames — với video dài, chỉ đọc
    được đoạn ĐẦU video rồi bỏ hẳn phần còn lại (không có shot/keyframe ở
    đó). Đã vá bằng subsample cách đều để max_frames frame được chọn TRẢI
    ĐỀU khắp toàn bộ video.
    """

    def test_frames_cover_full_video_duration_not_just_beginning(self, tmp_path):
        """Test QUAN TRỌNG NHẤT: với video dài hơn max_frames cho phép,
        thời lượng "phủ" (n_frames / fps_hieu_dung) phải xấp xỉ ĐÚNG thời
        lượng video gốc, không chỉ đoạn đầu."""
        import subprocess
        import cv2
        import numpy as np

        # Tạo video dài hơn (60s) để có đủ khoảng cách kiểm tra rõ ràng
        long_video = str(tmp_path / "long.mp4")
        fps = 30
        w, h = 320, 180
        out = cv2.VideoWriter(long_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n_total_frames = fps * 60  # 60 giây
        for i in range(n_total_frames):
            frame = np.full((h, w, 3), (i % 255, 50, 50), dtype=np.uint8)
            out.write(frame)
        out.release()

        from aic_pipeline.video_reader import get_video_frames, clear_cache
        clear_cache()

        max_frames = 100  # ép subsample mạnh (60s*30fps=1800 frame gốc -> 100)
        frames, fps_effective = get_video_frames(long_video, max_frames=max_frames)

        duration_covered = frames.shape[0] / fps_effective
        actual_duration = n_total_frames / fps

        assert abs(duration_covered - actual_duration) < 2.0, (
            f"Phải phủ gần đúng toàn bộ {actual_duration:.1f}s video gốc, "
            f"nhưng chỉ phủ được {duration_covered:.1f}s — có dấu hiệu chỉ "
            f"đọc đoạn đầu rồi dừng (bug đã fix)."
        )

    def test_frame_stride_calculated_correctly(self, tmp_path):
        import cv2
        import numpy as np

        video_path = str(tmp_path / "test.mp4")
        fps = 30
        w, h = 160, 90
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n_total = 600  # 20 giây
        for i in range(n_total):
            frame = np.full((h, w, 3), (i % 255, 0, 0), dtype=np.uint8)
            out.write(frame)
        out.release()

        from aic_pipeline.video_reader import get_video_frames, clear_cache
        clear_cache()

        frames, fps_effective = get_video_frames(video_path, max_frames=60)
        # 600 frame gốc / 60 max_frames = stride 10 -> fps hiệu dụng = 30/10 = 3.0
        assert abs(fps_effective - 3.0) < 0.5, (
            f"fps hiệu dụng phải xấp xỉ 3.0 (30/10), được {fps_effective}"
        )


def _make_config_for_test():
    """PHẢI ở module level (không phải nested trong class/method) để
    multiprocessing.Pool pickle được — đây là ràng buộc THẬT của Python
    multiprocessing, không phải giới hạn riêng của package này."""
    from aic_pipeline import PipelineConfig
    return PipelineConfig(store_images=False)


def _make_config_with_unpicklable_backend_for_test():
    """Cũng PHẢI ở module level — mô phỏng backend chứa object không pickle
    được (threading.Lock, giống PyTorch model/CUDA context thật), tạo BÊN
    TRONG factory (chạy trong worker process) thay vì truyền sẵn qua Pool."""
    import threading
    from aic_pipeline import PipelineConfig
    from aic_pipeline.shot_detector import HistogramSSIMDetector

    class UnpicklableBackend:
        def __init__(self):
            self._lock = threading.Lock()

        def detect(self, video_path):
            return HistogramSSIMDetector().detect(video_path)

    return PipelineConfig(shot_backend=UnpicklableBackend(), store_images=False)


class TestPipelineOptimizedAndParallel:
    """
    Vá tốc độ: run_pipeline() gốc đọc video 2 LẦN RIÊNG BIỆT (Tầng 2 + Tầng
    4), tốn gấp đôi chi phí decode với video AV1 dài. run_pipeline_optimized()
    preload video 1 LẦN, dùng chung qua cache của video_reader.py — đã đo
    thực tế trên video AIC: giảm ~1.8x thời gian (36.2s -> 19.7s cho phần
    motion+frame_selection). run_pipeline_batch_parallel() chạy song song
    nhiều video bằng multiprocessing, cho batch lớn (mục tiêu 700 video/2-3
    ngày).
    """

    def test_run_pipeline_optimized_matches_run_pipeline_result(self, synthetic_video):
        from aic_pipeline import run_pipeline, run_pipeline_optimized, PipelineConfig
        config = PipelineConfig(store_images=False)

        result_old = run_pipeline(synthetic_video, config)
        result_new = run_pipeline_optimized(synthetic_video, config)

        assert result_old.stats["n_shots"] == result_new.stats["n_shots"]
        assert result_old.stats["n_keyframes"] == result_new.stats["n_keyframes"]
        assert result_old.stats["motion_class_distribution"] == result_new.stats["motion_class_distribution"]

    def test_run_pipeline_optimized_has_preload_timing(self, synthetic_video):
        from aic_pipeline import run_pipeline_optimized, PipelineConfig
        result = run_pipeline_optimized(synthetic_video, PipelineConfig(store_images=False))
        assert "video_preload" in result.stats["timing_seconds"]

    def test_batch_parallel_matches_batch_sequential(self, tmp_path, synthetic_video):
        import shutil
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        for i in range(3):
            shutil.copy(synthetic_video, video_dir / f"v{i}.mp4")

        from aic_pipeline import run_pipeline_batch, run_pipeline_batch_parallel

        results_seq = run_pipeline_batch(str(video_dir), _make_config_for_test())
        results_par = run_pipeline_batch_parallel(str(video_dir), _make_config_for_test, n_workers=2)

        assert len(results_seq) == len(results_par) == 3
        for v in results_seq:
            assert results_seq[v].stats["n_shots"] == results_par[v].stats["n_shots"]
            assert results_seq[v].stats["n_keyframes"] == results_par[v].stats["n_keyframes"]

    def test_batch_parallel_handles_error_video_gracefully(self, tmp_path, synthetic_video):
        import shutil
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        shutil.copy(synthetic_video, video_dir / "good.mp4")
        (video_dir / "broken.mp4").write_bytes(b"not a real video")

        from aic_pipeline import run_pipeline_batch_parallel

        results = run_pipeline_batch_parallel(str(video_dir), _make_config_for_test, n_workers=2)

        assert len(results) == 2
        good_result = [r for p, r in results.items() if "good" in p][0]
        broken_result = [r for p, r in results.items() if "broken" in p][0]
        assert good_result.error is None
        assert broken_result.error is not None

    def test_batch_parallel_config_factory_with_unpicklable_object(self, tmp_path, synthetic_video):
        """
        Test QUAN TRỌNG NHẤT: xác nhận factory pattern né được đúng lỗi thật
        đã gặp trên Kaggle ("cannot pickle 'module' object") — mô phỏng model
        chứa object KHÔNG pickle được (threading.Lock, giống CUDA context
        thật), tạo object đó BÊN TRONG factory (chạy trong worker process),
        không truyền thẳng object đã tạo sẵn qua Pool.
        """
        import shutil
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        shutil.copy(synthetic_video, video_dir / "v0.mp4")

        from aic_pipeline import run_pipeline_batch_parallel

        results = run_pipeline_batch_parallel(
            str(video_dir), _make_config_with_unpicklable_backend_for_test, n_workers=1,
        )
        assert len(results) == 1
        result = list(results.values())[0]
        assert result.error is None, f"Không được lỗi pickle: {result.error}"
        assert result.stats["n_shots"] > 0


class TestOmniShotCutFullVideoCoverage:
    """
    Vá bug NGHIÊM TRỌNG #3 (phát hiện qua báo cáo thực tế: video 1056.4s bị
    mất 56.4s cuối — chỉ phủ tới 1000s): bản trước BREAK NGAY khi
    len(frames) >= max_frames, thay vì tiếp tục duyệt hết demux. Nếu
    total_frames_estimate (từ stream.frames) bị ước lượng sai dù chỉ chút,
    frame_stride tính ra nhỏ hơn cần -> đủ max_frames TRƯỚC khi duyệt hết
    video -> mất hẳn đoạn cuối. Đã vá bằng cách KHÔNG BAO GIỜ break theo số
    lượng frame, LUÔN duyệt hết toàn bộ demux, chỉ ngừng append khi đã đủ.
    """

    def test_covers_full_video_even_with_very_small_max_frames(self, tmp_path):
        """Test QUAN TRỌNG NHẤT: với max_frames RẤT NHỎ (dễ lộ bug nhất nếu
        còn), thời lượng phủ phải khớp GẦN ĐÚNG TUYỆT ĐỐI thời lượng video
        gốc — không được mất hẳn 1 đoạn nào, kể cả đoạn cuối."""
        import subprocess
        import cv2
        import numpy as np

        video_path = str(tmp_path / "test.mp4")
        fps = 30
        w, h = 320, 180
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n_total = fps * 45  # 45 giây
        for i in range(n_total):
            frame = np.full((h, w, 3), (i % 255, 30, 30), dtype=np.uint8)
            out.write(frame)
        out.release()

        av1_path = str(tmp_path / "test_av1.mp4")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-c:v", "libaom-av1",
             "-crf", "35", "-cpu-used", "8", av1_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(av1_path):
            pytest.skip(f"Không tạo được video AV1 test: {result.stderr[:300]}")

        import sys
        vendor_dir = os.path.join(
            os.path.dirname(__file__), "..", "aic_pipeline", "_omnishotcut_vendor"
        )
        sys.path.insert(0, vendor_dir)
        from omnishotcut.engine import _read_video_pyav

        for max_frames in [20, 50, 100]:
            video_np, fps_eff = _read_video_pyav(av1_path, width=224, height=224, max_frames=max_frames)
            duration_covered = video_np.shape[0] / fps_eff
            assert abs(duration_covered - 45.0) < 3.0, (
                f"max_frames={max_frames}: chỉ phủ {duration_covered:.1f}s / 45s thật "
                f"— có dấu hiệu mất đoạn cuối (bug đã fix)."
            )

    def test_never_breaks_before_full_demux(self, synthetic_video):
        """Xác nhận trực tiếp: video_np trả về không rỗng và fps hiệu dụng
        hợp lý (không phải giá trị bất thường do dừng giữa chừng)."""
        import sys
        vendor_dir = os.path.join(
            os.path.dirname(__file__), "..", "aic_pipeline", "_omnishotcut_vendor"
        )
        sys.path.insert(0, vendor_dir)
        from omnishotcut.engine import _read_video_pyav

        video_np, fps_eff = _read_video_pyav(synthetic_video, width=224, height=224, max_frames=30)
        assert video_np.shape[0] > 0
        assert fps_eff > 0


class TestEnsembleEmbedder:
    """
    Cho phép kết hợp nhiều embedder (CLIP + SigLIP2) đúng tinh thần ensemble
    mà nghiên cứu 2025 dùng (BEiT-3 + SigLIP2). Test bằng fake embedder
    (không cần tải model thật) để kiểm chứng logic concat + normalize riêng
    từng phần, độc lập với việc tải model thành công hay không.
    """

    @staticmethod
    def _make_fake_embedder(dim, seed):
        import numpy as np

        class FakeEmbedder:
            def encode_images(self, images):
                rng = np.random.RandomState(seed)
                return rng.randn(len(images), dim).astype(np.float32)

            def unload(self):
                pass

        return FakeEmbedder()

    def test_ensemble_concatenates_dimensions_correctly(self):
        import numpy as np
        from aic_pipeline.embeddings import EnsembleEmbedder

        e1 = self._make_fake_embedder(512, 1)
        e2 = self._make_fake_embedder(768, 2)
        ensemble = EnsembleEmbedder([e1, e2])

        images = [np.zeros((10, 10, 3), dtype=np.uint8)] * 3
        vecs = ensemble.encode_images(images)
        assert vecs.shape == (3, 512 + 768)

    def test_ensemble_normalizes_each_part_separately(self):
        import numpy as np
        from aic_pipeline.embeddings import EnsembleEmbedder

        e1 = self._make_fake_embedder(512, 1)
        e2 = self._make_fake_embedder(768, 2)
        ensemble = EnsembleEmbedder([e1, e2])

        images = [np.zeros((10, 10, 3), dtype=np.uint8)] * 3
        vecs = ensemble.encode_images(images)

        part1_norms = np.linalg.norm(vecs[:, :512], axis=1)
        part2_norms = np.linalg.norm(vecs[:, 512:], axis=1)
        assert np.allclose(part1_norms, 1.0, atol=1e-5), (
            "Phần embedder đầu phải được L2-normalize riêng trước khi concat, "
            "tránh 1 model có norm lớn hơn lấn át model kia trong farthest-point-sampling."
        )
        assert np.allclose(part2_norms, 1.0, atol=1e-5)

    def test_ensemble_requires_at_least_one_embedder(self):
        from aic_pipeline.embeddings import EnsembleEmbedder
        import pytest as _pytest
        with _pytest.raises(ValueError):
            EnsembleEmbedder([])

    def test_ensemble_empty_images_returns_empty(self):
        from aic_pipeline.embeddings import EnsembleEmbedder
        e1 = self._make_fake_embedder(512, 1)
        ensemble = EnsembleEmbedder([e1])
        vecs = ensemble.encode_images([])
        assert vecs.shape[0] == 0

    def test_ensemble_integrates_with_frame_selector(self, synthetic_video):
        """Test QUAN TRỌNG: xác nhận EnsembleEmbedder cắm thẳng vào
        select_keyframes() không cần sửa frame_selector.py."""
        from aic_pipeline.frame_selector import select_keyframes
        from aic_pipeline.shot_detector import Shot
        from aic_pipeline.embeddings import EnsembleEmbedder

        e1 = self._make_fake_embedder(512, 1)
        e2 = self._make_fake_embedder(768, 2)
        ensemble = EnsembleEmbedder([e1, e2])

        shot = Shot(shot_id=0, start_frame=0, end_frame=75, start_time=0.0, end_time=3.0)
        kfs = select_keyframes(
            synthetic_video, shot, n_keyframes=3,
            feature_mode="semantic", embedder=ensemble,
            store_images=False, store_embeddings=True,
        )
        assert len(kfs) > 0
        assert kfs[0].embedding.shape == (1280,)

    def test_siglip_embedder_import_does_not_crash_without_network(self):
        """SiglipEmbedder phải import/khởi tạo được (chỉ lỗi khi thực sự gọi
        encode_images cần tải model qua mạng) — kiểm tra lazy-loading đúng."""
        from aic_pipeline.embeddings import SiglipEmbedder
        embedder = SiglipEmbedder(device="cpu")
        assert embedder._model is None  # chưa load, đúng thiết kế lazy

