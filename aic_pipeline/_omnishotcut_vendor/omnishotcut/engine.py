'''
    This file is to inference arbitrary video files for Shot Cut
'''
import os, sys, shutil
import argparse
import numpy as np
import copy
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ĐÃ PATCH: thay decord bằng PyAV để đọc được codec AV1 (và các codec mới khác
# mà decord — bản cuối 0.6.0, dự án đã ngừng phát triển — không hỗ trợ).
# decord.VideoReader(path, width=W, height=H)[:].asnumpy() trả về (T,H,W,3) RGB
# đã resize sẵn; _read_video_pyav() bên dưới tái tạo đúng API/output tương đương
# để phần còn lại của file KHÔNG cần sửa gì thêm.
import av
import cv2
import logging

_logger_video_read = logging.getLogger("omnishotcut.video_read")


def _read_video_pyav(video_path, width, height, max_frames=6000, frame_stride=None):
    """
    Thay thế decord.VideoReader — trả về (video_np (T,H,W,3) RGB uint8, fps).

    ĐÃ VÁ LỖI TRÀN RAM: đọc TOÀN BỘ video vào 1 mảng numpy có thể tràn RAM với
    video dài (video AIC ~32.000 frame ở 1920x1080 -> ~4.9GB chỉ riêng mảng
    frame đã resize, cộng dồn với decode buffer/model dễ vượt RAM Kaggle,
    khiến kernel bị Kaggle tự restart do "tried to allocate more memory than
    is available"). Vá bằng 2 cơ chế:

      1. max_frames: giới hạn cứng số frame tối đa giữ trong RAM. Nếu video
         dài hơn, TỰ ĐỘNG lấy mẫu cách đều (uniform subsample) để giảm còn
         đúng max_frames, thay vì đọc hết rồi mới cắt (tránh đọc thừa).
         fps trả về được ĐIỀU CHỈNH TƯƠNG ỨNG để mọi mốc thời gian
         (Shot.start_time/end_time) vẫn tính đúng theo thời gian THẬT của
         video gốc, không bị lệch dù đã bỏ bớt frame.
      2. frame_stride: nếu bạn tự biết trước cần bỏ bớt bao nhiêu (ví dụ chỉ
         cần 1/2 số frame), truyền trực tiếp thay vì để hàm tự tính.

    Mặc định max_frames=6000 -> với 224x224x3 uint8 là ~900MB RAM, an toàn
    trên GPU notebook Kaggle tiêu chuẩn (13-16GB RAM khả dụng).
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps_original = float(stream.average_rate)
    total_frames_estimate = stream.frames or 0

    if frame_stride is None:
        if total_frames_estimate > max_frames:
            frame_stride = max(1, total_frames_estimate // max_frames)
        else:
            frame_stride = 1

    if frame_stride > 1:
        _logger_video_read.warning(
            f"Video có ~{total_frames_estimate} frame, vượt max_frames={max_frames} "
            f"-> lấy mẫu cách {frame_stride} frame để tránh tràn RAM. "
            f"fps hiệu dụng: {fps_original / frame_stride:.2f} (gốc: {fps_original:.2f})."
        )

    frames = []
    frame_idx = 0
    # Đọc theo PACKET (demux) rồi mới decode từng packet — thay vì generator
    # decode() có thể đã buffer/giải mã ngầm nhiều frame phía trước trước khi
    # Python kịp break, gây ra hàng loạt warning FFmpeg (đặc biệt rõ với AV1
    # phải fallback software decode) tiếp tục xuất hiện dù đã "break". Đọc
    # theo packet cho phép dừng NGAY sau khi đã đủ max_frames, không decode
    # dư thêm bất kỳ packet nào phía sau.
    stream.thread_type = "AUTO"  # cho phép FFmpeg dùng multi-thread decode nếu có, nhanh hơn
    stopped_early = False
    for packet in container.demux(video=0):
        for frame in packet.decode():
            if frame_idx % frame_stride == 0:
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3) RGB
                if width is not None and height is not None:
                    img = cv2.resize(img, (width, height))
                frames.append(img)
                if len(frames) >= max_frames:
                    stopped_early = True
                    break
            frame_idx += 1
        if stopped_early:
            _logger_video_read.warning(
                f"Đã đạt max_frames={max_frames}, dừng đọc sớm (video có thể dài hơn)."
            )
            break
    container.close()

    if not frames:
        raise ValueError(f"Không đọc được frame nào từ video: {video_path}")
    fps_effective = fps_original / frame_stride
    return np.stack(frames, axis=0), fps_effective


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from omnishotcut.architecture.backbone import build_backbone
from omnishotcut.architecture.transformer import build_transformer
from omnishotcut.architecture.model import OmniShotCut
from omnishotcut.datasets.transforms import Video_Augmentation_Transform
from omnishotcut.util.visualization import visualize_concated_frames
from omnishotcut.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping, intra_int2string, inter_int2string


# Video Transform
video_transform = Video_Augmentation_Transform(set_type = "val")




def load_model(checkpoint_path: str):


    # Check the checkpoint
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in state_dict or "model" not in state_dict:
        raise ValueError("Checkpoint must contain keys: 'args' and 'model'.")


    # Load the model
    model_args = state_dict["args"]
    backbone = build_backbone(model_args)
    transformer = build_transformer(model_args)
    model = OmniShotCut(
                            backbone,
                            transformer,
                            num_intra_relation_classes = model_args.num_intra_relation_classes,
                            num_inter_relation_classes = model_args.num_inter_relation_classes,
                            num_frames = model_args.max_process_window_length,
                            num_queries = model_args.num_queries,
                            aux_loss = model_args.aux_loss,
                        )
    model.load_state_dict(state_dict["model"], strict=True)
    pass  # device set bên ngoài bởi OmniShotCutDetector
    model.eval()


    return model, model_args



def split_videos(video, chunk_size, overlap_size):

    assert video.ndim == 4, "video must be (T, H, W, C)"
    assert overlap_size >= 0 and overlap_size < chunk_size

    T, H, W, C = video.shape
    stride = chunk_size - overlap_size

    # Form the return list
    return_list = []
    window_start_idx = 0

    while window_start_idx < T:

        window_end_idx = window_start_idx + chunk_size
        valid_len = min(chunk_size, T - window_start_idx)

        # Fetch current window
        chunk = video[window_start_idx:min(window_end_idx, T)]

        # Padding
        num_pad_frames = chunk_size - valid_len
        if num_pad_frames > 0:
            black = np.zeros((num_pad_frames, H, W, C), dtype=video.dtype)
            chunk = np.concatenate([chunk, black], axis=0)

        # Valid region for this window. We split the overlap region by half.
        left_overlap = overlap_size // 2
        right_overlap = overlap_size - left_overlap

        if window_start_idx == 0:
            valid_start_idx = 0
        else:
            valid_start_idx = window_start_idx + left_overlap

        if window_end_idx >= T:
            valid_end_idx = T
        else:
            valid_end_idx = window_end_idx - right_overlap

        return_list.append(
            [
                chunk,
                num_pad_frames,
                window_start_idx,
                valid_start_idx,
                valid_end_idx,
                valid_len,
            ]
        )

        # End
        if window_end_idx >= T:
            break

        window_start_idx += stride

    return return_list



def merge_predictions(pred_boundary_full, pred_boundary, duplicate_tolerance=2):

    # Sort
    pred_boundary = sorted(pred_boundary, key=lambda x: x["end_frame_idx"])

    # Merge
    for item in pred_boundary:

        # Check duplicate
        if len(pred_boundary_full) != 0:
            last_end_frame_idx = pred_boundary_full[-1]["end_frame_idx"]
            if abs(item["end_frame_idx"] - last_end_frame_idx) <= duplicate_tolerance:
                continue

        pred_boundary_full.append(item)

    return pred_boundary_full



def single_video_inference(video_path, model, model_args, overlap_window_length, max_frames=6000):


    # Init the parameter
    max_process_window_length = model_args.max_process_window_length
    process_height, process_width = model_args.process_height, model_args.process_width


    # Read the Video
    # ĐÃ VÁ: truyền max_frames để tránh tràn RAM với video dài (xem docstring
    # _read_video_pyav) — tham số mới, không có trong code gốc tác giả.
    video_np_full, fps = _read_video_pyav(
        video_path, width=process_width, height=process_height, max_frames=max_frames,
    )

    # Iterate all the clips
    pred_boundary_full = []

    for clip_idx, (video_np, num_pad_frames, window_start_idx, valid_start_idx, valid_end_idx, valid_len) in enumerate(split_videos(video_np_full, max_process_window_length, overlap_window_length)):

        # Transform
        video_tensor = video_transform(video_np).unsqueeze(0).to(next(model.parameters()).device)


        # Inference
        with torch.inference_mode():
            outputs = model(video_tensor)
        

        # Choose the label with max value
        probas_intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1] 
        probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]  
        range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]  
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)


        pred_boundary = []
        start_frame_idx_local = 0

        for keep_idx in range(len(query_intra_idx)):

            # Fetch Label
            pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())

            # Convert ranges from local window scale to video duration scale
            end_frame_idx_local = int(query_range_idx[keep_idx].detach().cpu())
            end_frame_idx_local = min(end_frame_idx_local, valid_len)

            pred_range = [start_frame_idx_local, end_frame_idx_local]
            pred_range_global = [
                window_start_idx + start_frame_idx_local,
                window_start_idx + end_frame_idx_local,
            ]

            # Sometimes model outputs the same start/end. Skip to avoid invalid range.
            if start_frame_idx_local >= end_frame_idx_local:
                continue

            # Append only the boundary inside the valid region
            end_frame_idx_global = window_start_idx + end_frame_idx_local

            if valid_start_idx < end_frame_idx_global <= valid_end_idx:
                pred_boundary.append(
                    {
                        "end_frame_idx": int(end_frame_idx_global),
                        "intra_label": int(pred_intra_label),
                        "inter_label": int(pred_inter_label),
                    }
                )

            start_frame_idx_local = end_frame_idx_local

            # End
            if end_frame_idx_local >= valid_len:
                break

        # Merge predicted results; here pred_boundary are already valid
        pred_boundary_full = merge_predictions(
            pred_boundary_full,
            pred_boundary,
        )


    # Convert boundary to range
    pred_ranges_full = []
    pred_intra_labels_full = []
    pred_inter_labels_full = []

    start_frame_idx_local = 0

    for item in pred_boundary_full:

        end_frame_idx = int(item["end_frame_idx"])

        if end_frame_idx <= start_frame_idx_local:
            continue

        pred_ranges_full.append(
            [
                int(start_frame_idx_local),
                int(end_frame_idx),
            ]
        )
        pred_intra_labels_full.append(int(item["intra_label"]))
        pred_inter_labels_full.append(int(item["inter_label"]))

        start_frame_idx_local = end_frame_idx


    return pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps



def _run_on_numpy(video_np, model, model_args, overlap_window_length):
    """Run inference on a pre-loaded numpy array (T, H, W, 3).
    Returns (ranges, intra_labels, inter_labels).
    """
    max_process_window_length = model_args.max_process_window_length

    pred_boundary_full = []

    for clip_idx, (video_chunk, num_pad_frames, window_start_idx, valid_start_idx, valid_end_idx, valid_len) in enumerate(split_videos(video_np, max_process_window_length, overlap_window_length)):

        video_tensor = video_transform(video_chunk).unsqueeze(0).to(next(model.parameters()).device)

        with torch.inference_mode():
            outputs = model(video_tensor)

        probas_intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1]
        probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]
        range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)

        pred_boundary = []
        start_frame_idx_local = 0

        for keep_idx in range(len(query_intra_idx)):

            pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())

            end_frame_idx_local = int(query_range_idx[keep_idx].detach().cpu())
            end_frame_idx_local = min(end_frame_idx_local, valid_len)

            if start_frame_idx_local >= end_frame_idx_local:
                continue

            end_frame_idx_global = window_start_idx + end_frame_idx_local

            if valid_start_idx < end_frame_idx_global <= valid_end_idx:
                pred_boundary.append(
                    {
                        "end_frame_idx": int(end_frame_idx_global),
                        "intra_label": int(pred_intra_label),
                        "inter_label": int(pred_inter_label),
                    }
                )

            start_frame_idx_local = end_frame_idx_local

            if end_frame_idx_local >= valid_len:
                break

        pred_boundary_full = merge_predictions(pred_boundary_full, pred_boundary)

    pred_ranges = []
    pred_intra_labels = []
    pred_inter_labels = []
    start_frame_idx = 0

    for item in pred_boundary_full:
        end_frame_idx = int(item["end_frame_idx"])
        if end_frame_idx <= start_frame_idx:
            continue
        pred_ranges.append([int(start_frame_idx), int(end_frame_idx)])
        pred_intra_labels.append(int(item["intra_label"]))
        pred_inter_labels.append(int(item["inter_label"]))
        start_frame_idx = end_frame_idx

    return pred_ranges, pred_intra_labels, pred_inter_labels


