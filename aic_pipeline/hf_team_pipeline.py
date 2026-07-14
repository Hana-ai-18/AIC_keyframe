"""
HF TEAM PIPELINE — chia việc cắt keyframe theo NHÓM VIDEO (prefix K01, K02...)
cho nhiều người làm song song, và tự động PUSH kết quả lên 1 Dataset chung
trên HuggingFace để cả nhóm gộp kết quả lại dễ dàng.

BỐI CẢNH: dataset nguồn enduong/AIC-video2025 (gated) có ~600+ video đặt tên
theo pattern K{2 chữ số}_V{3 chữ số} (ví dụ K01_V001.mp4, K01_V002.mp4, ...,
K02_V001.mp4, ...). Mỗi "K{XX}" là 1 nhóm ~20-30 video. Để nhiều người cắt
song song, chia việc theo nhóm K — mỗi người phụ trách 1 khoảng (ví dụ
K01-K05), không đụng nhau, không cần đồng bộ giữa các người khi đang chạy.

QUY TRÌNH:
  1. list_videos_by_prefix_range() — liệt kê đúng danh sách video của
     khoảng K được giao (ví dụ K01 đến K05), KHÔNG tải gì cả (chỉ list metadata,
     rất nhanh).
  2. stream_process_and_publish() — với TỪNG video trong danh sách: tải về
     -> chạy pipeline (cắt + lọc) -> lưu kết quả gọn (ảnh + metadata.json)
     -> XOÁ VIDEO GỐC -> PUSH kết quả gọn lên Dataset đích trên HuggingFace
     -> XOÁ kết quả local (đã push xong, không cần giữ nữa) -> chuyển video
     tiếp theo. Đĩa không bao giờ giữ nhiều hơn 1 video gốc + 1 kết quả gọn
     cùng lúc.
  3. Có cơ chế SKIP video đã xử lý xong (dựa vào việc kiểm tra file đã tồn
     tại trên Dataset đích trước khi tải) — cho phép NHIỀU NGƯỜI hoặc NHIỀU
     PHIÊN chạy cùng lúc/nối tiếp mà không làm trùng công.

CHUẨN BỊ (mỗi người tự làm 1 lần trên máy/Kaggle của mình):
  1. Vào https://huggingface.co/datasets/enduong/AIC-video2025, đăng nhập,
     bấm "Agree and access repository" (dataset NGUỒN, gated).
  2. Tạo Access Token tại https://huggingface.co/settings/tokens — quyền
     "Write" (không chỉ Read, vì cần PUSH lên dataset đích).
  3. Được thêm làm collaborator (quyền Write) vào dataset ĐÍCH
     hananguyen18/AIC_PixelPals (chủ dataset cần add từng thành viên qua
     Settings > Collaborators trên trang HuggingFace của dataset đó).
  4. Trên Kaggle: Add-ons > Secrets > thêm secret tên "HF_TOKEN".

CÁCH DÙNG (xem notebook mẫu đầy đủ):

    from aic_pipeline.hf_team_pipeline import (
        list_videos_by_prefix_range, stream_process_and_publish,
    )

    videos = list_videos_by_prefix_range(
        source_repo="enduong/AIC-video2025",
        prefix_start="K01", prefix_end="K05",
        hf_token=hf_token,
    )

    stream_process_and_publish(
        source_repo="enduong/AIC-video2025",
        target_repo="hananguyen18/AIC_PixelPals",
        video_files=videos,
        pipeline_config_factory=make_config_for_full_dataset,
        hf_token=hf_token,
    )
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import traceback
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("aic_pipeline.hf_team_pipeline")

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")

# Khớp đúng pattern K{2 chữ số}_V{3 chữ số} — ví dụ K01_V001.mp4, K23_V007.mp4
_PREFIX_PATTERN = re.compile(r"^(K\d{2})_V(\d{3})", re.IGNORECASE)


def _check_hf_hub_available():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise ImportError("Cần huggingface_hub: !pip install -q huggingface_hub")


def parse_prefix(filename: str) -> Optional[str]:
    """
    Trích prefix "K{XX}" từ tên file, ví dụ "K01_V001.mp4" -> "K01".
    Trả về None nếu tên file không khớp pattern (file không thuộc dataset
    video AIC chuẩn — ví dụ .gitattributes, README...).
    """
    basename = os.path.basename(filename)
    m = _PREFIX_PATTERN.match(basename)
    return m.group(1).upper() if m else None


def list_videos_by_prefix_range(
    source_repo: str,
    prefix_start: str,
    prefix_end: str,
    hf_token: Optional[str] = None,
) -> List[str]:
    """
    Liệt kê danh sách video trong khoảng prefix [prefix_start, prefix_end]
    (bao gồm cả 2 đầu), KHÔNG TẢI GÌ — chỉ list metadata qua HF API, rất
    nhanh dù dataset có hàng trăm/nghìn file.

    Args:
        source_repo: dataset nguồn, ví dụ "enduong/AIC-video2025".
        prefix_start, prefix_end: khoảng nhóm K được giao, ví dụ "K01", "K05"
                                   -> lấy tất cả video từ K01 đến K05
                                   (K01, K02, K03, K04, K05), không phân biệt
                                   hoa/thường, "k01" cũng hợp lệ.
        hf_token: token HF (bắt buộc vì dataset nguồn gated).

    Returns:
        List đường dẫn file trong repo nguồn (chưa tải), đã SẮP XẾP theo tên
        (K01_V001, K01_V002, ..., K05_V0XX) để xử lý theo thứ tự dễ theo dõi.

    Raises:
        ValueError nếu prefix_start > prefix_end (khoảng rỗng) — báo lỗi rõ
        ràng ngay từ đầu, tránh chạy xong mới phát hiện không có video nào.
    """
    _check_hf_hub_available()
    from huggingface_hub import HfApi

    prefix_start = prefix_start.upper()
    prefix_end = prefix_end.upper()
    if prefix_start > prefix_end:
        raise ValueError(
            f"prefix_start ({prefix_start}) phải <= prefix_end ({prefix_end}). "
            f"Ví dụ đúng: prefix_start='K01', prefix_end='K05'."
        )

    api = HfApi(token=hf_token)
    all_files = api.list_repo_files(repo_id=source_repo, repo_type="dataset")

    selected = []
    for f in all_files:
        if not f.lower().endswith(VIDEO_EXTENSIONS):
            continue
        prefix = parse_prefix(f)
        if prefix is None:
            logger.debug(f"Bỏ qua file không khớp pattern K{{XX}}_V{{XXX}}: {f}")
            continue
        if prefix_start <= prefix <= prefix_end:
            selected.append(f)

    selected.sort()
    logger.info(
        f"Tìm thấy {len(selected)} video trong khoảng [{prefix_start}, {prefix_end}] "
        f"trên tổng {len(all_files)} file của repo {source_repo}."
    )
    if not selected:
        logger.warning(
            f"KHÔNG tìm thấy video nào trong khoảng [{prefix_start}, {prefix_end}] "
            f"— kiểm tra lại đúng prefix chưa (xem danh sách file thật trên "
            f"https://huggingface.co/datasets/{source_repo}/tree/main)."
        )
    return selected


def _download_one_video(source_repo: str, filename: str, local_dir: str, hf_token: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=source_repo, filename=filename, repo_type="dataset",
        local_dir=local_dir, token=hf_token,
    )


def _remote_result_exists(target_repo: str, video_id: str, hf_token: Optional[str]) -> bool:
    """
    Kiểm tra kết quả của video_id ĐÃ được push lên target_repo chưa — dùng để
    SKIP video đã xử lý, cho phép nhiều người/nhiều phiên chạy nối tiếp mà
    không làm trùng công. Kiểm tra bằng cách tìm file
    "<video_id>/metadata.json" trong danh sách file remote.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        all_files = api.list_repo_files(repo_id=target_repo, repo_type="dataset")
    except Exception as e:
        logger.warning(f"Không kiểm tra được file đã tồn tại trên {target_repo}: {e}")
        return False
    marker = f"{video_id}/metadata.json"
    return marker in all_files


def _save_result_compact(result, output_dir: str, video_id: str) -> Dict:
    """Lưu kết quả gọn: ảnh keyframe .jpg + metadata.json — KHÔNG lưu video gốc."""
    video_out_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_out_dir, exist_ok=True)

    keyframe_meta = []
    for i, kf in enumerate(result.keyframes):
        entry = {
            "index": i, "shot_id": kf.shot_id, "frame_index": kf.frame_index,
            "timestamp": round(kf.timestamp, 3), "sharpness": round(kf.sharpness, 2),
            "feature_mode": kf.feature_mode,
        }
        if kf.image is not None:
            img_name = f"kf_{i:04d}_shot{kf.shot_id}_t{kf.timestamp:.2f}.jpg"
            cv2.imwrite(
                os.path.join(video_out_dir, img_name), kf.image,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            entry["image_file"] = img_name
        keyframe_meta.append(entry)

    meta = {
        "video_id": video_id,
        "stats": result.stats,
        "keyframes": keyframe_meta,
    }
    with open(os.path.join(video_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def _push_result_to_hf(
    target_repo: str, video_out_dir: str, video_id: str, hf_token: Optional[str],
):
    """
    Push toàn bộ thư mục kết quả gọn (ảnh + metadata.json) của 1 video lên
    dataset đích, đặt vào đúng thư mục con "<video_id>/" trên đó — dùng
    upload_folder (1 commit duy nhất cho cả thư mục, hiệu quả hơn từng file).
    """
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_folder(
        repo_id=target_repo,
        folder_path=video_out_dir,
        path_in_repo=video_id,
        repo_type="dataset",
        commit_message=f"Thêm keyframe: {video_id}",
    )


def _append_result_to_zip(zip_path: str, video_out_dir: str, video_id: str):
    """
    ĐÃ THÊM — CƠ CHẾ DỰ PHÒNG ZIP: ghi kết quả của 1 video vào file zip cục
    bộ, ĐỘC LẬP HOÀN TOÀN với việc push HuggingFace có thành công hay không.

    Vì sao cần: nếu HuggingFace bị lỗi (token sai quyền, gated repo, rate
    limit, mất mạng giữa chừng...) — như đã gặp thật (403 Forbidden) — kết
    quả đã xử lý xong KHÔNG ĐƯỢC MẤT. File zip là nguồn sự thật thứ 2, luôn
    được ghi trước khi thử push lên HF, để dù HF lỗi bao nhiêu lần, bạn vẫn
    có đủ kết quả để tải về hoặc tự push lại sau.

    Dùng chế độ APPEND ("a") — mỗi video được thêm vào ngay khi xử lý xong,
    không cần đợi tới cuối mới nén 1 lần (nếu Kaggle session hết giờ giữa
    chừng, zip vẫn có đủ các video đã xử lý TRƯỚC thời điểm đó).
    """
    import zipfile

    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(video_out_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                arcname = os.path.join(video_id, fname)
                zf.write(full_path, arcname)


def stream_process_and_publish(
    source_repo: str,
    target_repo: Optional[str],
    video_files: List[str],
    pipeline_config_factory: Callable,
    hf_token: Optional[str] = None,
    skip_existing: bool = True,
    tmp_download_dir: str = "/kaggle/working/_tmp_video_download",
    tmp_output_dir: str = "/kaggle/working/_tmp_processed",
    use_optimized: bool = True,
    zip_backup_path: Optional[str] = "/kaggle/working/keyframes_backup.zip",
    push_to_hf: bool = True,
) -> Dict[str, dict]:
    """
    Entry point chính: với TỪNG video trong video_files — tải về, chạy
    pipeline, lưu kết quả gọn, GHI VÀO ZIP DỰ PHÒNG (luôn luôn), rồi CỐ GẮNG
    push lên target_repo (nếu push_to_hf=True — lỗi push KHÔNG làm mất kết
    quả vì đã có trong zip), xoá sạch local tạm, chuyển video tiếp theo.

    Đĩa không bao giờ giữ quá 1 video gốc + 1 kết quả gọn tạm cùng lúc (ngoại
    trừ file zip dự phòng tích luỹ dần — đây là kết quả CẦN GIỮ LẠI, không
    phải rác tạm).

    ĐÃ THÊM CƠ CHẾ DỰ PHÒNG ZIP (theo yêu cầu: "tránh trường hợp huggingface
    bị lỗi"): mỗi video xử lý xong LUÔN được ghi vào zip_backup_path TRƯỚC,
    sau đó mới thử push lên HuggingFace. Nếu push lỗi (như 403 Forbidden đã
    gặp thật, hoặc mất mạng, rate limit...), kết quả VẪN AN TOÀN trong zip —
    không phải chạy lại từ đầu. Đặt push_to_hf=False để CHỈ lưu zip, không
    thử push HF (hữu ích nếu biết trước token có vấn đề, xử lý xong rồi push
    thủ công sau).

    Args:
        source_repo: dataset nguồn video, ví dụ "enduong/AIC-video2025".
        target_repo: dataset ĐÍCH lưu kết quả, ví dụ "hananguyen18/AIC_PixelPals".
                     Có thể để None nếu push_to_hf=False (chỉ dùng zip).
        video_files: danh sách file cần xử lý — lấy từ
                     list_videos_by_prefix_range().
        pipeline_config_factory: hàm KHÔNG THAM SỐ trả về PipelineConfig mới.
        hf_token: token HF — cần quyền Read trên source_repo, và Write trên
                     target_repo nếu push_to_hf=True.
        skip_existing: True (mặc định) — bỏ qua video ĐÃ CÓ TRONG ZIP DỰ
                     PHÒNG (kiểm tra local, không cần mạng) HOẶC đã có trên
                     target_repo (nếu push_to_hf=True) — cho phép chạy lại
                     giữa chừng mà không xử lý trùng.
        zip_backup_path: đường dẫn file zip dự phòng — None để TẮT hẳn cơ
                     chế zip (không khuyến nghị, chỉ dùng nếu chắc chắn HF
                     luôn hoạt động ổn định).
        push_to_hf: True (mặc định) — cố gắng push lên target_repo sau khi
                     đã ghi zip. False — CHỈ ghi zip, không đụng gì tới HF
                     (dùng khi biết trước có vấn đề với HF, xử lý toàn bộ
                     bằng zip rồi push thủ công sau khi khắc phục).

    Returns:
        Dict {video_id: metadata_dict} cho các video xử lý THÀNH CÔNG (đã
        ghi vào zip) trong lần chạy này — bao gồm cả video mà bước PUSH HF
        bị lỗi (miễn zip ghi thành công), vì mục tiêu cốt lõi là "xử lý xong,
        không mất dữ liệu", còn push HF là bước cộng thêm.
    """
    _check_hf_hub_available()

    if push_to_hf and not target_repo:
        raise ValueError("push_to_hf=True yêu cầu phải có target_repo.")

    os.makedirs(tmp_download_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)
    if zip_backup_path:
        os.makedirs(os.path.dirname(zip_backup_path) or ".", exist_ok=True)

    def _already_in_zip(video_id: str) -> bool:
        if not zip_backup_path or not os.path.exists(zip_backup_path):
            return False
        import zipfile
        try:
            with zipfile.ZipFile(zip_backup_path, "r") as zf:
                return any(name.startswith(f"{video_id}/metadata.json") for name in zf.namelist())
        except zipfile.BadZipFile:
            return False

    results: Dict[str, dict] = {}
    n_ok, n_err, n_skip, n_push_failed = 0, 0, 0, 0
    t_start = time.time()

    for i, filename in enumerate(video_files):
        video_id = os.path.splitext(os.path.basename(filename))[0]

        if skip_existing:
            if _already_in_zip(video_id):
                logger.info(f"[{i+1}/{len(video_files)}] Bỏ qua (đã có trong zip dự phòng): {video_id}")
                n_skip += 1
                continue
            if push_to_hf and _remote_result_exists(target_repo, video_id, hf_token):
                logger.info(f"[{i+1}/{len(video_files)}] Bỏ qua (đã có trên {target_repo}): {video_id}")
                n_skip += 1
                continue

        logger.info(f"[{i+1}/{len(video_files)}] Tải: {filename}")
        t0 = time.time()
        local_video_path = None
        video_out_dir = os.path.join(tmp_output_dir, video_id)

        try:
            local_video_path = _download_one_video(source_repo, filename, tmp_download_dir, hf_token)
            t_download = time.time() - t0
            logger.info(
                f"  Đã tải ({t_download:.1f}s, {os.path.getsize(local_video_path)/1e6:.1f}MB) "
                f"-> chạy pipeline..."
            )

            config = pipeline_config_factory()
            from .pipeline import run_pipeline, run_pipeline_optimized
            fn = run_pipeline_optimized if use_optimized else run_pipeline

            t1 = time.time()
            result = fn(local_video_path, config)
            t_process = time.time() - t1

            meta = _save_result_compact(result, tmp_output_dir, video_id)

            # GHI VÀO ZIP DỰ PHÒNG TRƯỚC — đây là bước ĐẢM BẢO KHÔNG MẤT DỮ
            # LIỆU, làm TRƯỚC khi thử push HF (có thể lỗi).
            t_zip = 0.0
            if zip_backup_path:
                t2 = time.time()
                _append_result_to_zip(zip_backup_path, video_out_dir, video_id)
                t_zip = time.time() - t2

            results[video_id] = meta
            n_ok += 1

            # CỐ GẮNG PUSH LÊN HF — lỗi ở đây KHÔNG làm video này bị coi là
            # thất bại (đã có trong zip rồi), chỉ log cảnh báo riêng.
            t_push = 0.0
            if push_to_hf:
                try:
                    t3 = time.time()
                    _push_result_to_hf(target_repo, video_out_dir, video_id, hf_token)
                    t_push = time.time() - t3
                except Exception as e:
                    n_push_failed += 1
                    logger.warning(
                        f"  Push lên HuggingFace THẤT BẠI cho {video_id} (dữ liệu vẫn AN "
                        f"TOÀN trong {zip_backup_path}): {type(e).__name__}: {e}"
                    )

            elapsed = time.time() - t_start
            avg = elapsed / (n_ok + n_skip if (n_ok + n_skip) else 1)
            remaining = avg * (len(video_files) - i - 1)
            logger.info(
                f"  -> {result.stats['n_shots']} shot, {result.stats['n_keyframes']} keyframe | "
                f"xử lý {t_process:.1f}s, zip {t_zip:.1f}s, push {t_push:.1f}s | "
                f"ước tính còn {remaining/3600:.1f}h cho {len(video_files)-i-1} video"
            )

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"  LỖI khi xử lý {filename}: {err_msg}")
            n_err += 1

        finally:
            if local_video_path is not None and os.path.exists(local_video_path):
                os.remove(local_video_path)
            if os.path.exists(video_out_dir):
                shutil.rmtree(video_out_dir, ignore_errors=True)

    try:
        shutil.rmtree(tmp_download_dir, ignore_errors=True)
        shutil.rmtree(tmp_output_dir, ignore_errors=True)
    except Exception:
        pass

    total_elapsed = time.time() - t_start
    logger.info(
        f"HOÀN TẤT: {n_ok} thành công (đã lưu zip), {n_push_failed} lỗi push HF "
        f"(vẫn an toàn trong zip), {n_err} lỗi xử lý, {n_skip} bỏ qua "
        f"/ tổng {len(video_files)} video. Thời gian: {total_elapsed/3600:.2f}h."
    )
    if zip_backup_path and os.path.exists(zip_backup_path):
        logger.info(
            f"File zip dự phòng: {zip_backup_path} "
            f"({os.path.getsize(zip_backup_path)/1e6:.1f} MB)"
        )
    return results


def republish_zip_to_hf(
    zip_backup_path: str,
    target_repo: str,
    hf_token: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[str, bool]:
    """
    HÀM DỰ PHÒNG BỔ SUNG: push lại TOÀN BỘ nội dung file zip dự phòng lên
    HuggingFace — dùng khi lần chạy trước push HF bị lỗi (ví dụ 403 do
    token sai quyền), sau khi đã khắc phục vấn đề token, chạy hàm này để
    đẩy nốt kết quả đã có sẵn trong zip lên mà KHÔNG cần tải/xử lý lại video
    từ đầu.

    Args:
        zip_backup_path: đường dẫn file zip đã tạo bởi stream_process_and_publish.
        target_repo: dataset đích, ví dụ "hananguyen18/AIC_PixelPals".
        hf_token: token HF có quyền Write trên target_repo.
        skip_existing: bỏ qua video ĐÃ CÓ trên target_repo rồi (kiểm tra qua
                     API), tránh push trùng nếu 1 phần zip đã được push
                     thành công từ trước.

    Returns:
        Dict {video_id: True/False} — True nếu push thành công, False nếu lỗi.
    """
    _check_hf_hub_available()
    import zipfile
    import tempfile

    results: Dict[str, bool] = {}

    with zipfile.ZipFile(zip_backup_path, "r") as zf:
        all_names = zf.namelist()
        video_ids = sorted(set(name.split("/")[0] for name in all_names if "/" in name))
        logger.info(f"File zip có kết quả của {len(video_ids)} video: {video_ids}")

        for i, video_id in enumerate(video_ids):
            if skip_existing and _remote_result_exists(target_repo, video_id, hf_token):
                logger.info(f"[{i+1}/{len(video_ids)}] Bỏ qua (đã có trên {target_repo}): {video_id}")
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                video_out_dir = os.path.join(tmp_dir, video_id)
                os.makedirs(video_out_dir, exist_ok=True)
                member_names = [n for n in all_names if n.startswith(f"{video_id}/")]
                for name in member_names:
                    zf.extract(name, tmp_dir)

                try:
                    _push_result_to_hf(target_repo, video_out_dir, video_id, hf_token)
                    results[video_id] = True
                    logger.info(f"[{i+1}/{len(video_ids)}] Push THÀNH CÔNG: {video_id}")
                except Exception as e:
                    results[video_id] = False
                    logger.error(f"[{i+1}/{len(video_ids)}] Push LỖI: {video_id}: {e}")

    n_ok = sum(results.values())
    logger.info(f"Hoàn tất republish: {n_ok}/{len(results)} video push thành công.")
    return results
