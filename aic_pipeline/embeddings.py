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


class SiglipEmbedder:
    """
    Wrapper nạp SigLIP2 qua HuggingFace `transformers` — thay thế cho CLIP khi
    cần độ chính xác ngữ nghĩa cao hơn. Đã tra cứu benchmark thật trước khi
    viết class này (không đoán): SigLIP2 vượt CLIP/SigLIP gốc rõ rệt trên các
    benchmark retrieval chuẩn (COCO R@1: 47.4%->53.2%, ImageNet zero-shot:
    76.7%->79.1%), và trong nghiên cứu 2025 mới nhất được dùng NGANG HÀNG với
    BEiT-3 trong hệ thống ensemble — không có bằng chứng BEiT-3 vượt trội hẳn
    SigLIP2. Khác BEiT-3 (cần clone repo riêng, tự tải checkpoint .pth + file
    .spm, cài deepspeed/apex), SigLIP2 có sẵn qua transformers chuẩn, không
    cần setup phức tạp — cùng mức rủi ro thấp như CLIP.

    Cùng interface encode_images() với ClipEmbedder — dùng thay thế trực tiếp,
    không cần sửa gì ở frame_selector.py.

    Cách dùng:
        embedder = SiglipEmbedder(device="cuda")
        vecs = embedder.encode_images([frame1_bgr, frame2_bgr, ...])
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: str = "cuda",
        batch_size: int = 64,
    ):
        if not _check_torch_available():
            raise ImportError(
                "SiglipEmbedder cần torch + transformers (bản mới, hỗ trợ SigLIP2). "
                "Trên Kaggle: !pip install -q --upgrade transformers"
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
        from transformers import AutoModel, AutoProcessor

        logger.info(f"Đang nạp model {self.model_name} lên {self.device}...")
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._torch = torch
        logger.info("Nạp model xong.")

    def encode_images(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """Cùng chữ ký với ClipEmbedder.encode_images() — xem docstring ở đó."""
        if not images_bgr:
            return np.zeros((0, 768), dtype=np.float32)  # siglip2-base có dim 768

        self._lazy_load()
        import cv2
        torch = self._torch

        all_vecs = []
        for i in range(0, len(images_bgr), self.batch_size):
            batch = images_bgr[i:i + self.batch_size]
            batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch]
            inputs = self._processor(images=batch_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_vecs.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(all_vecs, axis=0)

    def unload(self):
        if self._model is not None:
            del self._model, self._processor
            self._model = None
            if _check_torch_available():
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class EnsembleEmbedder:
    """
    Kết hợp NHIỀU embedder cùng lúc (ví dụ CLIP + SigLIP2) bằng cách CONCAT
    vector embedding — đúng tinh thần ensemble mà nghiên cứu 2025 dùng (kết
    hợp BEiT-3 + SigLIP2 cho retrieval).

    LƯU Ý VỀ CHI PHÍ (đã bàn kỹ trước khi viết): dùng ensemble ở Tầng 4 (lọc
    keyframe TRONG 1 shot ngắn) tốn GẤP ĐÔI thời gian encode so với 1 model,
    trong khi mục đích chỉ là so sánh vài frame gần nhau — lợi ích không rõ
    ràng bằng chi phí bỏ ra. CHỈ NÊN DÙNG khi:
      1. Nghi ngờ 1 model bỏ sót khác biệt ngữ nghĩa tinh vi (ví dụ shot có
         nhiều người/vật giống nhau, cần độ nhạy cao hơn).
      2. Đã thử 1 model trước, thấy kết quả chọn frame chưa đủ tốt khi xem
         bằng mắt.
    Mặc định của pipeline vẫn nên dùng 1 embedder (CLIP hoặc SigLIP2) — chỉ
    chuyển sang EnsembleEmbedder cho các shot/video cụ thể cần độ chính xác
    cao hơn, không bật tràn lan cho toàn bộ dataset lớn (tốn thời gian không
    tương xứng khi cần xử lý 700 video trong 2-3 ngày).

    Cách dùng:
        from aic_pipeline.embeddings import ClipEmbedder, SiglipEmbedder, EnsembleEmbedder
        embedder = EnsembleEmbedder([
            ClipEmbedder(device="cuda"),
            SiglipEmbedder(device="cuda"),
        ])
        vecs = embedder.encode_images([frame1_bgr, ...])  # vecs.shape = (N, dim1+dim2)
    """

    def __init__(self, embedders: List[object]):
        if not embedders:
            raise ValueError("EnsembleEmbedder cần ít nhất 1 embedder.")
        self.embedders = embedders

    def encode_images(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Trả về concat embedding của tất cả embedder con, MỖI PHẦN ĐÃ ĐƯỢC
        L2-NORMALIZE RIÊNG trước khi concat (đảm bảo mỗi model đóng góp công
        bằng vào khoảng cách Euclidean dùng trong farthest-point-sampling,
        không để 1 model có norm lớn hơn lấn át model kia).
        """
        if not images_bgr:
            return np.zeros((0, 0), dtype=np.float32)

        parts = []
        for emb in self.embedders:
            vecs = emb.encode_images(images_bgr)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs_normalized = vecs / (norms + 1e-8)
            parts.append(vecs_normalized)
        return np.concatenate(parts, axis=1)

    def unload(self):
        for emb in self.embedders:
            if hasattr(emb, "unload"):
                emb.unload()
