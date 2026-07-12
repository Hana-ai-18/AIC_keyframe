"""
QUERY-AWARE RERANKING BẰNG MLLM (chỉ gọi lúc có query thật — khác WESp caption
offline tràn lan). Logic không đổi so với bản trước, chỉ cập nhật import cho
khớp cấu trúc package mới.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Protocol

from .frame_selector import Keyframe
from .motion_scorer import MotionProfile


@dataclasses.dataclass
class QueryAwareScore:
    keyframe: Keyframe
    relevance_score: float
    caption: Optional[str] = None


class MLLMScorerBackend(Protocol):
    """
    Giao diện chuẩn cho MLLM client (Gemini/GPT-4o...). Tự implement class này.

    Ví dụ khung code cho Gemini trên Kaggle (cần internet bật trong notebook
    settings, và API key lưu trong Kaggle Secrets):

        import google.generativeai as genai
        from kaggle_secrets import UserSecretsClient

        class GeminiScorer:
            def __init__(self):
                secrets = UserSecretsClient()
                api_key = secrets.get_secret("GEMINI_API_KEY")
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-2.0-flash")

            def score(self, image_bgr, query: str) -> tuple[float, str]:
                import cv2
                from PIL import Image
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                prompt = (
                    f'Ảnh này có khớp với mô tả sau không: "{query}"? '
                    f'Trả lời CHỈ bằng JSON: {{"score": <0-1>, "caption": "<mô tả ngắn>"}}'
                )
                response = self.model.generate_content([prompt, pil_img])
                import json, re
                text = response.text.strip()
                text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
                data = json.loads(text)
                return float(data["score"]), data.get("caption", "")
    """

    def score(self, image_bgr, query: str) -> tuple: ...


def rerank_candidates(
    candidates: List[Keyframe],
    motion_profiles: List[MotionProfile],
    query: str,
    mllm: MLLMScorerBackend,
    dynamic_only: bool = True,
    top_k_to_score: int = 20,
) -> List[QueryAwareScore]:
    """Entry point — xem giải thích chiến lược đầy đủ trong docstring các lượt trước."""
    motion_by_shot = {mp.shot_id: mp for mp in motion_profiles}
    to_score: List[Keyframe] = []
    skipped: List[Keyframe] = []

    for kf in candidates[:top_k_to_score]:
        mp = motion_by_shot.get(kf.shot_id)
        is_dynamic = mp is not None and mp.motion_class == "dynamic"
        if dynamic_only and not is_dynamic:
            skipped.append(kf)
        else:
            to_score.append(kf)

    results: List[QueryAwareScore] = []
    for kf in to_score:
        if kf.image is None:
            raise ValueError(
                f"Keyframe (shot={kf.shot_id}, frame={kf.frame_index}) không có "
                f"ảnh (image=None) — cần store_images=True ở Tầng 4."
            )
        score, caption = mllm.score(kf.image, query)
        results.append(QueryAwareScore(keyframe=kf, relevance_score=score, caption=caption))

    for kf in skipped:
        results.append(QueryAwareScore(keyframe=kf, relevance_score=-1.0, caption=None))

    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results
