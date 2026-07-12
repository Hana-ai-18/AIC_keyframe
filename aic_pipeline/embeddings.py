"""
EMBEDDINGS — wrapper nạp model CLIP/SigLIP2/BEiT-3 qua HuggingFace, dùng GPU nếu có.

Đây là phần bổ sung so với bản trước: vá lỗ hổng "Tầng 4 chỉ dùng đặc trưng màu/cạnh
rẻ, không có ngữ nghĩa" — giờ có tuỳ chọn dùng embedding CLIP thật (chạy tốt trên
A100 của Kaggle) để đo độ khác biệt NGỮ NGHĨA giữa các frame, không chỉ khác biệt
thị giác thấp.

Thiết kế lazy-import: module này KHÔNG bắt buộc phải có torch/transformers để
import — chỉ raise lỗi rõ ràng khi bạn thực sự gọi .encode_images() mà chưa cài.
Điều này giúp phần còn lại của pipeline (Tầng 1-3, và Tầng 4 ở chế độ "rẻ")
vẫn chạy được trên máy không có GPU/torch.

MẶC ĐỊNH DÙNG: openai/clip-vit-base-patch32 — nhẹ, tải nhanh, đủ tốt để đo
độ khác biệt ngữ nghĩa giữa các frame trong 1 shot (không cần model SOTA nặng
ở bước LỌC keyframe — model nặng hơn (BEiT-3/SigLIP2-giant) nên để dành cho
bước EMBEDDING CHÍNH THỨC lúc build index retrieval, là việc khác với việc lọc
keyframe ở đây).
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger("aic_pipeline.embeddings")

_TORCH_AVAILABLE: Optional[bool] = None


def _check_torch_available() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


class ClipEmbedder:
    """
    Wrapper nạp CLIP qua HuggingFace `transformers`, encode ảnh BGR (OpenCV) thành
    vector embedding L2-normalized.

    Cách dùng trên Kaggle (đã có torch + GPU sẵn):
        embedder = ClipEmbedder(model_name="openai/clip-vit-base-patch32", device="cuda")
        vecs = embedder.encode_images([frame1_bgr, frame2_bgr, ...])
        # vecs.shape = (N, D), đã L2-normalize -> dùng cosine similarity = dot product

    Nếu muốn dùng model mạnh hơn (SigLIP2, BEiT-3 qua timm/open_clip), tự thay
    _load_model()/encode_images() — interface bên ngoài (encode_images) giữ nguyên
    để không phải sửa frame_selector.py.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        batch_size: int = 64,
    ):
        if not _check_torch_available():
            raise ImportError(
                "ClipEmbedder cần torch + transformers. Trên Kaggle, cài bằng:\n"
                "  !pip install -q transformers torch --upgrade\n"
                "(Kaggle notebook GPU thường đã cài sẵn torch — nếu vậy chỉ cần "
                "  !pip install -q transformers)"
            )
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Đang nạp model {self.model_name} lên {self.device}...")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._torch = torch
        logger.info("Nạp model xong.")

    def encode_images(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Args:
            images_bgr: danh sách ảnh BGR (numpy, dtype uint8) — định dạng OpenCV chuẩn.
        Returns:
            np.ndarray shape (N, D), đã L2-normalize từng vector.
        """
        if not images_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        self._lazy_load()
        import cv2
        torch = self._torch

        all_vecs = []
        for i in range(0, len(images_bgr), self.batch_size):
            batch = images_bgr[i:i + self.batch_size]
            # CLIP cần ảnh RGB
            batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch]
            inputs = self._processor(images=batch_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_vecs.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(all_vecs, axis=0)

    def unload(self):
        """Giải phóng GPU memory — gọi sau khi xử lý xong 1 batch video lớn nếu cần."""
        if self._model is not None:
            del self._model, self._processor
            self._model = None
            if _check_torch_available():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
