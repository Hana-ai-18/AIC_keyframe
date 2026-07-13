import logging
import os
import numpy as np
import torch
import av
import cv2
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import enable_progress_bars

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

from omnishotcut.engine import load_model, _run_on_numpy
from omnishotcut.label_correspondence import unique_intra_label_mapping, intra_int2string, inter_int2string


_DEFAULT_HF_FILENAME = "OmniShotCut_ckpt.pth"


def _read_video_pyav_for_wrapper(video_path, width, height, max_frames=6000):
    """
    ĐÃ PATCH: thay decord.VideoReader bằng PyAV để đọc được codec AV1 (và các
    codec mới khác mà decord — bản cuối 0.6.0, dự án đã ngừng phát triển —
    không hỗ trợ). Gọi lại đúng hàm engine._read_video_pyav (đã vá lỗi tràn
    RAM với video dài — xem docstring ở đó) để tránh trùng logic 2 nơi.
    """
    from omnishotcut.engine import _read_video_pyav
    video_np, _fps = _read_video_pyav(video_path, width=width, height=height, max_frames=max_frames)
    return video_np


class OmniShotCutModel:

    def __init__(self, model, model_args):
        self._model = model
        self._model_args = model_args

    def inference(self, video, mode="clean_shot", overlap=20):
        """Run shot cut detection on a video.

        Args:
            video: str file path | np.ndarray (T,H,W,3) uint8 RGB | torch.Tensor (T,H,W,3)
            mode: "clean_shot" — general cuts only (no transitions)
                  "default"    — all detected shots with full labels
            overlap: number of overlap frames between adjacent inference windows

        Returns:
            ranges:       list of [start_frame, end_frame]
            intra_labels: list of int (0=General, 1=Dissolve, 2=Wipes, ...)
            inter_labels: list of int (0=New_Start, 1=Hard_Cut, 2=Transition_Source, ...)
        """
        if isinstance(video, str):
            h, w = self._model_args.process_height, self._model_args.process_width
            video_np = _read_video_pyav_for_wrapper(video, width=w, height=h)
        elif isinstance(video, torch.Tensor):
            if video.ndim != 4 or video.shape[-1] != 3:
                raise ValueError(f"Tensor must be (T, H, W, 3), got {tuple(video.shape)}")
            video_np = video.cpu().numpy()
            if video_np.dtype != np.uint8:
                if video_np.min() < 0.0 or video_np.max() > 1.0:
                    raise ValueError(f"Float tensor must be in [0, 1], got range [{video_np.min():.3f}, {video_np.max():.3f}]")
                video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = np.asarray(video)
            if video_np.ndim != 4 or video_np.shape[-1] != 3:
                raise ValueError(f"numpy array must be (T, H, W, 3), got {video_np.shape}")
            if video_np.dtype != np.uint8:
                if video_np.min() < 0.0 or video_np.max() > 1.0:
                    raise ValueError(f"Float array must be in [0, 1], got range [{video_np.min():.3f}, {video_np.max():.3f}]")
                video_np = (video_np * 255).astype(np.uint8)

        ranges, intra_labels, inter_labels = _run_on_numpy(video_np, self._model, self._model_args, overlap)

        if mode == "clean_shot":
            general_idx = unique_intra_label_mapping["general"]
            keep = [i for i, lbl in enumerate(intra_labels) if lbl == general_idx]
            ranges = np.array(ranges)[keep].tolist() if keep else []
            return ranges

        intra_labels = [intra_int2string.get(x, str(x)) for x in intra_labels]
        inter_labels = [inter_int2string.get(x, str(x)) for x in inter_labels]
        return ranges, intra_labels, inter_labels


def load(checkpoint_path, filename=_DEFAULT_HF_FILENAME):
    """Load model weights and return an OmniShotCutModel instance.

    checkpoint_path can be:
      - a local file path  → load directly (filename ignored)
      - a HF repo ID       → download the specified filename from that repo
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading checkpoint from HuggingFace: {checkpoint_path} ...")
        enable_progress_bars()
        checkpoint_path = hf_hub_download(repo_id=checkpoint_path, filename=filename)

    logger.info(f"Loading OmniShotCut from {checkpoint_path} ...")
    model, model_args = load_model(checkpoint_path)
    logger.info("OmniShotCut loaded successfully.")
    return OmniShotCutModel(model, model_args)
