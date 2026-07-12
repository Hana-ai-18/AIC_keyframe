"""
HF STREAMING RUNNER — xử lý dataset video LỚN (162GB+) trên Kaggle mà không
cần tải hết cùng lúc.

Nguyên tắc: TẢI 1 VIDEO -> CHẠY PIPELINE -> LƯU KẾT QUẢ NHỎ (keyframe .jpg +
embedding .npy + metadata .json) -> XOÁ VIDEO GỐC -> LẶP LẠI. Tại mọi thời
điểm, đĩa chỉ giữ đúng 1 video gốc (~vài trăm MB) thay vì cả 162GB.

YÊU CẦU:
  - Dataset enduong/AIC-video2025 trên HuggingFace là GATED (cần đăng nhập +
    chấp nhận điều kiện). Trước khi chạy, bạn cần:
      1. Vào https://huggingface.co/datasets/enduong/AIC-video2025, đăng nhập,
         bấm "Agree and access repository".
      2. Tạo Access Token tại https://huggingface.co/settings/tokens (quyền
         "Read" là đủ).
      3. Trên Kaggle: Add-ons > Secrets > thêm secret tên "HF_TOKEN" với giá
         trị là token vừa tạo.
  - pip install huggingface_hub

CÁCH DÙNG (xem thêm ví dụ đầy đủ trong notebook):

    from aic_pipeline.hf_streaming import stream_process_hf_dataset
    from aic_pipeline.streaming_batch import FastPipelineConfig

    config = FastPipelineConfig(feature_mode="semantic", embedder=embedder, ...)
    stream_process_hf_dataset(
        repo_id="enduong/AIC-video2025",
        output_dir="/kaggle/working/processed",
        pipeline_config=config,
        hf_token=hf_token,   # lấy từ Kaggle Secrets
        limit=10,             # thử trước vài video
    )
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .streaming_batch import FastPipelineConfig, FastPipelineResult, run_pipeline_fast

logger = logging.getLogger("aic_pipeline.hf_streaming")

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def _check_hf_hub_available():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise ImportError(
            "Cần cài huggingface_hub: !pip install -q huggingface_hub"
        )


def list_video_files_in_repo(repo_id: str, hf_token: Optional[str] = None) -> List[str]:
    """
    Liệt kê TẤT CẢ đường dẫn file video trong 1 dataset repo trên HuggingFace,
    KHÔNG TẢI GÌ CẢ — chỉ list metadata (rất nhanh, không tốn băng thông).
    """
    _check_hf_hub_available()
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    video_files = [f for f in all_files if f.lower().endswith(VIDEO_EXTENSIONS)]
    logger.info(f"Tìm thấy {len(video_files)} file video trong repo {repo_id} "
                f"(tổng {len(all_files)} file).")
    return video_files


def _download_one_video(
    repo_id: str, filename: str, local_dir: str, hf_token: Optional[str]
) -> str:
    """Tải ĐÚNG 1 file video, trả về đường dẫn local."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset",
        local_dir=local_dir, token=hf_token,
    )
    return local_path


def _save_result_compact(
    result: FastPipelineResult, output_dir: str, video_id: str
) -> Dict:
    """
    Lưu kết quả GỌN NHẸ ra đĩa: ảnh keyframe (.jpg, nén), embedding (.npy nếu
    có), và metadata (.json) — KHÔNG lưu video gốc. Trả về dict tóm tắt.
    """
    video_out_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_out_dir, exist_ok=True)

    keyframe_meta = []
    embeddings_list = []

    for i, kf in enumerate(result.keyframes):
        entry = {
            "index": i, "shot_id": kf.shot_id, "frame_index": kf.frame_index,
            "timestamp": kf.timestamp, "sharpness": kf.sharpness,
            "feature_mode": kf.feature_mode,
        }
        if kf.image is not None:
            img_name = f"kf_{i:04d}_shot{kf.shot_id}_t{kf.timestamp:.2f}.jpg"
            cv2.imwrite(
                os.path.join(video_out_dir, img_name), kf.image,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            entry["image_file"] = img_name
        if kf.embedding is not None:
            embeddings_list.append(kf.embedding)
        keyframe_meta.append(entry)

    if embeddings_list:
        np.save(os.path.join(video_out_dir, "embeddings.npy"), np.stack(embeddings_list))

    meta = {
        "video_id": video_id,
        "video_path_original": result.video_path,
        "stats": result.stats,
        "keyframes": keyframe_meta,
    }
    with open(os.path.join(video_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def stream_process_hf_dataset(
    repo_id: str,
    output_dir: str,
    pipeline_config: Optional[FastPipelineConfig] = None,
    hf_token: Optional[str] = None,
    limit: Optional[int] = None,
    skip_existing: bool = True,
    on_video_done: Optional[Callable[[str, Dict], None]] = None,
    tmp_download_dir: str = "/kaggle/working/_tmp_video_download",
) -> Dict[str, dict]:
    """
    Entry point chính: xử lý TOÀN BỘ (hoặc `limit` video đầu) của 1 dataset
    HuggingFace theo kiểu STREAMING — tải từng video, xử lý, lưu kết quả gọn,
    XOÁ VIDEO GỐC, lặp lại. Không bao giờ giữ quá 1 video gốc trên đĩa.

    Args:
        repo_id: ví dụ "enduong/AIC-video2025".
        output_dir: nơi lưu kết quả gọn (keyframe .jpg + embedding .npy +
                    metadata.json), ví dụ "/kaggle/working/processed".
        pipeline_config: FastPipelineConfig — cấu hình pipeline 4 tầng.
        hf_token: token HF (cần vì dataset gated) — lấy từ Kaggle Secrets,
                  xem hướng dẫn ở đầu file.
        limit: giới hạn số video xử lý (dùng để test trước).
        skip_existing: True thì bỏ qua video đã có metadata.json trong
                       output_dir (cho phép DỪNG VÀ CHẠY LẠI giữa chừng mà
                       không xử lý trùng — quan trọng với dataset lớn, có thể
                       Kaggle session hết giờ giữa chừng).
        on_video_done: callback tuỳ chọn, gọi sau mỗi video xử lý xong,
                       nhận (video_id, metadata_dict) — dùng để đẩy dữ liệu
                       vào Milvus/Elasticsearch ngay lập tức nếu muốn.
        tmp_download_dir: thư mục tạm để tải video vào trước khi xử lý —
                          LUÔN bị dọn sau mỗi video, không tích luỹ.

    Returns:
        Dict {video_id: metadata_dict} cho các video xử lý THÀNH CÔNG trong
        lần chạy này (không bao gồm video đã skip vì skip_existing).
    """
    _check_hf_hub_available()
    pipeline_config = pipeline_config or FastPipelineConfig()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_download_dir, exist_ok=True)

    video_files = list_video_files_in_repo(repo_id, hf_token)
    if limit is not None:
        video_files = video_files[:limit]

    results: Dict[str, dict] = {}
    n_ok, n_err, n_skip = 0, 0, 0

    for i, filename in enumerate(video_files):
        video_id = os.path.splitext(os.path.basename(filename))[0]
        video_out_dir = os.path.join(output_dir, video_id)
        meta_path = os.path.join(video_out_dir, "metadata.json")

        if skip_existing and os.path.exists(meta_path):
            logger.info(f"[{i+1}/{len(video_files)}] Bỏ qua (đã xử lý): {video_id}")
            n_skip += 1
            continue

        logger.info(f"[{i+1}/{len(video_files)}] Tải: {filename}")
        t0 = time.time()
        local_path = None
        try:
            local_path = _download_one_video(repo_id, filename, tmp_download_dir, hf_token)
            t_download = time.time() - t0

            logger.info(f"  Đã tải ({t_download:.1f}s, "
                        f"{os.path.getsize(local_path) / 1e6:.1f}MB) -> chạy pipeline...")

            t1 = time.time()
            result = run_pipeline_fast(local_path, pipeline_config)
            t_process = time.time() - t1

            meta = _save_result_compact(result, output_dir, video_id)
            results[video_id] = meta

            logger.info(
                f"  -> {result.stats['n_shots']} shot, {result.stats['n_keyframes']} "
                f"keyframe, xử lý {t_process:.1f}s"
            )
            n_ok += 1

            if on_video_done is not None:
                on_video_done(video_id, meta)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"  LỖI khi xử lý {filename}: {err_msg}")
            n_err += 1

        finally:
            # LUÔN xoá video gốc sau khi xử lý (dù thành công hay lỗi) —
            # đây là bước cốt lõi đảm bảo đĩa không bao giờ đầy.
            if local_path is not None and os.path.exists(local_path):
                os.remove(local_path)

    try:
        shutil.rmtree(tmp_download_dir, ignore_errors=True)
    except Exception:
        pass

    logger.info(
        f"HOÀN TẤT: {n_ok} thành công, {n_err} lỗi, {n_skip} bỏ qua "
        f"(đã xử lý trước đó) / tổng {len(video_files)} video."
    )
    return results
