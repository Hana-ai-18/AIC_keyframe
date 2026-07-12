"""
Demo chạy pipeline Tầng 1-4 trên 1 video, in kết quả chi tiết, lưu ảnh keyframe.

Chạy: python3 demo.py sample_data/test_video.mp4
      python3 demo.py sample_data/test_video.mp4 --semantic   (cần torch+GPU)
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from aic_pipeline import PipelineConfig, run_pipeline


def main(video_path: str, out_dir: str = "outputs", use_semantic: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    embedder = None
    if use_semantic:
        from aic_pipeline.embeddings import ClipEmbedder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Nạp CLIP embedder trên device={device}...")
        embedder = ClipEmbedder(device=device)

    config = PipelineConfig(
        feature_mode="semantic" if use_semantic else "cheap",
        embedder=embedder,
        static_threshold=0.5,
        dynamic_threshold=2.0,
        store_images=True,
    )

    result = run_pipeline(video_path, config)

    print("=" * 70)
    print(f"VIDEO: {video_path}  |  feature_mode={config.feature_mode}")
    print("=" * 70)

    print(f"\n[Tầng 1] Số shot phát hiện: {len(result.shots)}")
    for s in result.shots:
        print(f"  Shot {s.shot_id}: {s.start_time:.2f}s-{s.end_time:.2f}s "
              f"({s.duration:.2f}s), boundary={s.boundary_type}")

    print(f"\n[Tầng 2] Motion profile:")
    for mp in result.motion_profiles:
        print(f"  Shot {mp.shot_id}: mean_flow={mp.mean_flow_magnitude:.3f} "
              f"-> class='{mp.motion_class}'")

    print(f"\n[Tầng 3] Ngân sách keyframe:")
    for b in result.shot_budgets:
        print(f"  Shot {b.shot_id}: {b.n_keyframes} keyframe | {b.reason}")

    print(f"\n[Tầng 4] Tổng keyframe: {len(result.keyframes)}")
    for kf in result.keyframes:
        print(f"  Shot {kf.shot_id} | t={kf.timestamp:.2f}s | "
              f"sharpness={kf.sharpness:.1f} | mode={kf.feature_mode}")
        if kf.image is not None:
            fname = f"{out_dir}/shot{kf.shot_id}_t{kf.timestamp:.2f}s.jpg"
            cv2.imwrite(fname, kf.image)

    print("\n" + "=" * 70)
    print("THỐNG KÊ")
    print("=" * 70)
    print(json.dumps(result.stats, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu {len(result.keyframes)} ảnh vào: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="sample_data/test_video.mp4")
    parser.add_argument("--semantic", action="store_true", help="Dùng CLIP embedding (cần torch+GPU)")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    main(args.video, args.out, args.semantic)
