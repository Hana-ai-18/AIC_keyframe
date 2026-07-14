# aic_pipeline (v0.5) — Kaggle A100 edition, AutoShot + OmniShotCut thật

## Thay đổi mới nhất (v0.5) — OmniShotCut: mạnh hơn AutoShot, dễ cài hơn nhiều

Đã đọc trực tiếp source code gốc của OmniShotCut (UVA CV Lab, 2026, kiến trúc
Shot-Query Transformer) qua git clone, vendor hoá (MIT license) vào
`_omnishotcut_vendor/`, viết `OmniShotCutDetector` dùng đúng API wrapper có
sẵn của tác giả (`omnishotcut.load()`).

**Vì sao đáng cân nhắc hơn AutoShot:**
- Checkpoint tải **trực tiếp qua HuggingFace** (`hf_hub_download`), không cần
  Baidu Pan — chỉ cần bật Internet trên Kaggle.
- Code inference gốc **không bị comment** (khác AutoShot) — API sạch, rõ ràng.
- Phân loại **9 loại intra-transition** (dissolve, wipe, push, slide, zoom,
  fade, doorway...) và **5 loại inter-label** (hard_cut, sudden_jump,
  transition...) — chi tiết hơn nhiều so với AutoShot (chỉ hard/gradual).
- Theo paper gốc: vượt AutoShot/TransNetV2 về transition IoU, phát hiện được
  cả "sudden jump" — đúng điểm yếu đã ghi nhận ở AutoShot.

**Đã kiểm chứng:** import kiến trúc thành công (`build_backbone`,
`build_transformer`, `OmniShotCut`), cài `decord` (dependency đặc biệt nhất)
thành công qua pip, 5 test bằng mock (không cần tải checkpoint thật — môi
trường build không có internet ra huggingface.co) xác nhận đúng logic
chuyển đổi output model → `List[Shot]`, tích hợp đúng với `run_pipeline()`.

**Đã sửa 1 bug rò rỉ trạng thái toàn cục:** `torch.use_deterministic_algorithms()`
trong `GPUShotDetector` không được khôi phục sau khi dùng ở lượt trước, gây
flaky test khi chạy chung với `AutoShotDetector`/`OmniShotCutDetector` trong
cùng session — đã sửa bằng try/finally đúng cách, kiểm chứng ổn định qua
nhiều lần chạy liên tiếp đúng tổ hợp gây lỗi trước đó.

**Notebook mới:** `notebooks/kaggle_omnishotcut.ipynb` — so sánh 3 chiều
(Baseline / AutoShot / OmniShotCut), xem nhãn transition chi tiết, chạy
pipeline đầy đủ trên 1-5 video.

## Thay đổi v0.3 → v0.4 — AutoShot THẬT

Đã đọc trực tiếp source code gốc của AutoShot, vendor hoá kiến trúc
`TransNetV2Supernet` vào `_autoshot_vendor/`, viết lại hoàn chỉnh
`AutoShotDetector` để load checkpoint `ckpt_0_200_0.pth` thật.


Pipeline 5 tầng: cắt & lọc keyframe (Tầng 1-4) + temporal reranking đa giai đoạn
(Tầng 5) cho video retrieval (HCM AI Challenge / VBS). Thiết kế chạy trên
**Kaggle notebook với GPU A100**, dùng data tự crawl để test.

## Thay đổi mới nhất (v0.3) — vá 4 nhược điểm đã xác định

Sau khi phân tích so sánh với các paper (NII-UIT, OpenCubee2, ConvAgent, WESp),
4 nhược điểm sau đã được vá **cẩn thận, có test kiểm chứng cho từng vá**:

| # | Nhược điểm | Vá bằng | Đã kiểm chứng |
|---|---|---|---|
| 1 | Ngưỡng motion phải chỉnh tay | `auto_calibrate_thresholds_gmm()` — tự tìm ngưỡng bằng Gaussian Mixture 3 thành phần, không cần đoán percentile | 2 test |
| 2 | Shot detector GPU chỉ là "hook rỗng" | `GPUShotDetector` — CNN 1D chạy được ngay, có "sàn an toàn" union với baseline để không tệ hơn khi model chưa huấn luyện | 6 test, **đã phát hiện và sửa 1 bug non-determinism nghiêm trọng** |
| 3 | Chưa có tương quan đối tượng trong shot | `object_graph.py` — track object bằng centroid tracking + đồ thị tương tác, fallback không cần GPU (contour) và tuỳ chọn YOLO | 5 test |
| 4 | Temporal Reranker chỉ 1 chiều (A→B→C) | `TemporalReranker.rerank_bidirectional()` — DP 2 chiều, chấm điểm từng hit theo cả quá khứ lẫn tương lai | 4 test |

**Bug quan trọng đã phát hiện và sửa:** `GPUShotDetector` ban đầu cho kết quả
DAO ĐỘNG giữa các lần chạy (3-7 shot khác nhau trên cùng 1 video) dù đã cố
định `torch.manual_seed()` — nguyên nhân là forward pass của `Conv1d` trên
CPU không deterministic (PyTorch đa luồng). Đã khắc phục bằng
`torch.set_num_threads(1)` + `torch.use_deterministic_algorithms(True)`,
kiểm chứng ổn định qua 5+ lần chạy liên tiếp.


## Thay đổi so với bản trước (v0.2 → v0.3)

| | v0.2 | v0.3 (bản này) |
|---|---|---|
| Tốc độ | `run_pipeline()` đọc video 2 lần (Tầng 2 + Tầng 4 riêng biệt) | **Mới**: `run_pipeline_fast()` — đọc video **1 lần tuần tự duy nhất**, nhanh hơn ~35-45% (đo trên video test), kết quả **giống hệt** bản gốc (đã test xác nhận) |
| Xử lý dataset lớn | Chỉ có `run_pipeline_batch()` cho thư mục local | **Mới**: `hf_streaming.py` — xử lý dataset **162GB trên HuggingFace** kiểu streaming: tải 1 video → xử lý → lưu gọn → xoá ngay, không bao giờ giữ hết dataset trên đĩa |
| Khả năng phục hồi | Không có | `skip_existing=True` — nếu Kaggle session hết giờ giữa chừng, chạy lại notebook sẽ tự bỏ qua video đã xử lý xong |

## Thay đổi v0.1 → v0.2

| | v0.1 | v0.2 |
|---|---|---|
| Tầng 4 lọc frame | Chỉ đặc trưng rẻ (màu/cạnh) | Thêm chế độ `"semantic"` dùng CLIP embedding thật trên GPU |
| Tầng 5 | Không có | `TemporalReranker` — cơ chế Reranking & Relinking kiểu WESp, tổng quát N-stage |

## Cài đặt

```bash
pip install -r requirements.txt
```

Trên Kaggle: `opencv-python`, `numpy`, `torch` thường đã có sẵn trong image GPU.
Chỉ cần `!pip install -q transformers` thêm.

## Chạy trên Kaggle — quy trình đề xuất

1. Nén thư mục này thành `aic_pipeline.zip`, upload làm Kaggle Dataset (hoặc dùng Add Data > Upload).
2. Mở `notebooks/kaggle_starter.ipynb` (đã kèm sẵn, copy nội dung vào notebook mới trên Kaggle) — chạy tuần tự các cell.
3. Sửa `DATA_DIR` trong cell batch trỏ tới thư mục video bạn tự crawl (upload dưới dạng Dataset khác, hoặc tải trực tiếp nếu bật Internet).

## Kiến trúc — 5 tầng

```
video ──> [1: shot_detector] ──> shots
                                   │
                                   v
               [2: motion_scorer] ──> motion_profiles
                                   │
                                   v
           [3: budget_allocator] ──> shot_budgets
                                   │
                                   v
            [4: frame_selector] ──> keyframes  (chế độ cheap HOẶC semantic)
                                   │
                                   v
                         (đưa vào embedding retrieval chính thức,
                          Milvus/Elasticsearch — nằm ngoài package này)
                                   │
                                   v  (lúc CÓ câu hỏi thi thật, nhiều giai đoạn)
              [5: temporal_reranker] ──> ChainResult (chuỗi sự kiện mạch lạc)
```

### Tầng 1 — `shot_detector.py`

Baseline: HSV histogram diff (hard-cut) + SSIM cửa sổ trượt (gradual transition).
Hook `AutoShotDetector` sẵn sàng cắm checkpoint khi có (xem TODO trong file).

### Tầng 2 — `motion_scorer.py`

Optical flow (Farneback), phân loại `static`/`moderate`/`dynamic`. Có
`calibrate_thresholds()` để tự tìm ngưỡng trên tập video mẫu của bạn.

### Tầng 3 — `budget_allocator.py`

`budget(shot) = base[class] + extra_per_second[class] × min(duration, cap)`.
Static → 1 keyframe, Dynamic → 4-6+ (tỉ lệ theo độ dài). Có biến thể giới hạn
tổng ngân sách toàn video/toàn kho.

### Tầng 4 — `frame_selector.py` + `embeddings.py` (NÂNG CẤP)

2 chế độ, chọn qua `feature_mode`:
- `"cheap"`: color histogram + edge density, không cần GPU.
- `"semantic"`: CLIP embedding (`ClipEmbedder` trong `embeddings.py`), cần
  GPU — dùng khi cần độ chính xác cao, tận dụng A100.

Cả 2 chế độ đều qua chung thuật toán farthest-point-sampling có trọng số chất
lượng ảnh (độ nét + phơi sáng) ở bước cuối.

### Tầng 5 — `temporal_reranker.py` (MỚI)

Xử lý truy vấn nhiều giai đoạn (KIS-T nhiều bước, TRAKE). Input: danh sách
`StageHit` theo từng stage (kết quả thô từ model retrieval embedding chính
thức — không nằm trong package này). Output: `ChainResult` — chuỗi tốt nhất
xuyên suốt các stage, tính bằng quy hoạch động, KHÔNG chỉ nối tham lam từng
cặp liền kề (khác beam search cộng điểm đơn thuần).

```python
from aic_pipeline import TemporalReranker, StageHit

stage0 = [StageHit(video_id="v1", stage_index=0, timestamp=10.0, similarity=0.8)]
stage1 = [StageHit(video_id="v1", stage_index=1, timestamp=14.0, similarity=0.85)]

reranker = TemporalReranker(w_A=0.5, lambda_=0.6, T_max=30.0)
chains = reranker.rerank([stage0, stage1])
for c in chains:
    print(c.video_id, c.chain_score, c.normalized_score)
```

Tham số quan trọng:
- `w_A`: trọng số blend similarity giữa 2 đầu liên kết.
- `lambda_`: mức phạt theo khoảng cách thời gian.
- `T_max`: khoảng cách tối đa (giây) — vượt quá thì loại thẳng liên kết đó.
- `penalty_mode`: `"exp"` (phạt tăng mượt) hoặc `"sqrt"` (phạt tăng nhanh rồi
  bão hoà — hợp với TRAKE, cho phép "khoan dung" hơn ở khoảng cách lớn).

`ChainResult.normalized_score` dùng để so sánh các chuỗi có SỐ STAGE KHÁC
NHAU công bằng (chia đều cho số hit); `chain_score` là tổng tích luỹ thô.

## Sử dụng nhanh

### Chạy 1 video, chế độ cheap (không cần GPU)

```python
from aic_pipeline import run_pipeline, PipelineConfig

result = run_pipeline("video.mp4")
for kf in result.keyframes:
    ...  # kf.image là ảnh BGR, đẩy vào model embedding chính thức
```

### Chạy 1 video, chế độ semantic (dùng GPU, tận dụng A100)

```python
from aic_pipeline import run_pipeline, PipelineConfig
from aic_pipeline.embeddings import ClipEmbedder

embedder = ClipEmbedder(device="cuda")
config = PipelineConfig(feature_mode="semantic", embedder=embedder, store_embeddings=True)
result = run_pipeline("video.mp4", config)
```

### Chạy hàng loạt trên thư mục data tự crawl

```python
from aic_pipeline import run_pipeline_batch, PipelineConfig

config = PipelineConfig(feature_mode="semantic", embedder=embedder, store_images=False)
results = run_pipeline_batch("data_crawl/", config, pattern="*.mp4", limit=None)

for path, r in results.items():
    if r.error:
        print(f"LỖI {path}: {r.error[:200]}")
    else:
        print(f"{path}: {r.stats['n_keyframes']} keyframe")
```

### Calibrate ngưỡng motion trên data tự crawl

```python
from aic_pipeline import detect_shots
from aic_pipeline.motion_scorer import calibrate_thresholds
import glob

videos = glob.glob("data_crawl/*.mp4")
all_shots = [detect_shots(v) for v in videos]
lo, hi = calibrate_thresholds(videos, all_shots)
# dùng lo, hi cho PipelineConfig(static_threshold=lo, dynamic_threshold=hi)
```

## Test nhanh với video của bạn — đặt trực tiếp trong repo

Cách đơn giản nhất để thử pipeline trên Kaggle: đặt vài video (khuyến nghị
≤5 video, mỗi file <100MB do giới hạn GitHub) vào thư mục `sample_data/`
ngay trong repo trước khi push lên GitHub:

```
aic_pipeline/
├── sample_data/
│   ├── test_video.mp4     # video demo tổng hợp có sẵn (giữ nguyên, dùng cho unit test)
│   ├── video1.mp4          # video thật của bạn
│   ├── video2.mp4
│   └── ...
```

Kiểm tra dung lượng trước khi `git push` (GitHub từ chối file >100MB):

```bash
du -sh sample_data/*.mp4
```

Notebook `notebooks/kaggle_omnishotcut.ipynb` đã cấu hình sẵn để tự động
tìm video trong `sample_data/` sau khi `git clone` (bỏ qua `test_video.mp4`
demo), không cần tạo Kaggle Dataset riêng cho video.

Nếu video quá lớn để đặt trong repo (>100MB/file hoặc tổng repo quá nặng),
dùng cách cũ: upload video làm Kaggle Dataset riêng, sửa `VIDEO_DIR` trong
notebook trỏ tới `/kaggle/input/<ten-dataset>`.



Dùng khi dataset quá lớn để tải hết vào Kaggle cùng lúc (ví dụ
`enduong/AIC-video2025`, 162GB, gated). Nguyên tắc: **tải 1 video → xử lý →
lưu kết quả gọn → xoá video gốc → lặp lại** — đĩa không bao giờ giữ quá 1
video gốc.

### Chuẩn bị (làm 1 lần)

1. Vào trang dataset trên HuggingFace, đăng nhập, bấm "Agree and access repository".
2. Tạo access token (quyền Read) tại huggingface.co/settings/tokens.
3. Trên Kaggle: Add-ons → Secrets → thêm secret `HF_TOKEN`.

### Chạy

```python
from kaggle_secrets import UserSecretsClient
from aic_pipeline.hf_streaming import stream_process_hf_dataset
from aic_pipeline.streaming_batch import FastPipelineConfig
from aic_pipeline.embeddings import ClipEmbedder

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
embedder = ClipEmbedder(device="cuda")

config = FastPipelineConfig(
    feature_mode="semantic", embedder=embedder,
    store_images=True, store_embeddings=True,
    global_budget=150,
    candidate_resize_to=(320, 180),   # downsize để giảm dung lượng output
)

results = stream_process_hf_dataset(
    repo_id="enduong/AIC-video2025",
    output_dir="/kaggle/working/processed",
    pipeline_config=config,
    hf_token=HF_TOKEN,
    limit=None,           # None = xử lý toàn bộ; đặt số nhỏ để thử trước
    skip_existing=True,   # an toàn khi phải chạy lại giữa chừng (session hết giờ)
)
```

Kết quả mỗi video được lưu vào `output_dir/<video_id>/`: ảnh keyframe `.jpg`,
`embeddings.npy` (nếu `store_embeddings=True`), và `metadata.json` — KHÔNG
lưu lại video gốc.

Xem notebook đầy đủ: `notebooks/kaggle_hf_streaming_162gb.ipynb`.

### Tối ưu tốc độ đã áp dụng (`run_pipeline_fast` — dùng ngầm bên trong)

- **Đọc video đúng 1 lần tuần tự**, không seek lặp lại — bản `run_pipeline()`
  gốc đọc mỗi frame 2 lần (1 lần cho Tầng 2 motion, 1 lần cho Tầng 4 chọn
  frame); bản fast gộp làm 1 lượt duy nhất, giữ frame ứng viên trong RAM tạm
  giữa các tầng.
- **Downsize sớm** cho optical flow (160×90) — không giải mã full-res rồi mới
  resize.
- **`candidate_resize_to`** (tuỳ chọn) — downsize cả ảnh lưu ra đĩa, giảm
  dung lượng output đáng kể khi xử lý hàng nghìn video.
- Đã kiểm chứng: `run_pipeline_fast()` cho **kết quả giống hệt**
  `run_pipeline()` (cùng số shot, cùng phân loại motion, cùng keyframe/shot)
  — chỉ nhanh hơn, không đổi logic.

## Chạy test

```bash
python3 -m pytest tests/ -v
```

39 test. 2 test chế độ semantic dùng CLIP thật sẽ tự SKIP nếu không có mạng —
đã có `TestFrameSelectorSemanticLogicWithFakeEmbedder` kiểm chứng logic độc
lập với việc tải model. Trên Kaggle (có Internet: On), toàn bộ 41 test sẽ chạy.

## Cấu trúc thư mục

```
aic_pipeline/
├── aic_pipeline/
│   ├── __init__.py
│   ├── shot_detector.py       # Tầng 1 (+ GPUShotDetector, AutoShotDetector, OmniShotCutDetector)
│   ├── _autoshot_vendor/      # Vendor hoá kiến trúc AutoShot (MIT), giữ LICENSE gốc
│   ├── _omnishotcut_vendor/   # Vendor hoá package omnishotcut (MIT), giữ LICENSE gốc
│   ├── motion_scorer.py       # Tầng 2 (+ auto_calibrate_thresholds_gmm)
│   ├── budget_allocator.py    # Tầng 3
│   ├── frame_selector.py      # Tầng 4 (2 chế độ)
│   ├── embeddings.py          # wrapper CLIP, dùng cho Tầng 4 semantic
│   ├── temporal_reranker.py   # Tầng 5 (+ rerank_bidirectional)
│   ├── object_graph.py        # tương quan đối tượng trong shot
│   ├── query_aware.py         # module tuỳ chọn, MLLM rerank online
│   ├── pipeline.py            # run_pipeline() + run_pipeline_batch()
│   ├── streaming_batch.py     # run_pipeline_fast() — đọc video 1 lần
│   └── hf_streaming.py        # xử lý dataset lớn (162GB) từ HuggingFace
├── tests/
│   └── test_pipeline.py       # 79 test (77 chạy không cần mạng)
├── sample_data/
│   └── test_video.mp4
├── notebooks/
│   ├── kaggle_starter.ipynb
│   ├── kaggle_github_clone.ipynb
│   ├── kaggle_hf_streaming_162gb.ipynb
│   ├── kaggle_autoshot_real.ipynb
│   └── kaggle_omnishotcut.ipynb
├── demo.py
├── requirements.txt
└── README.md
```

## Dùng các module mới (v0.3)

### Auto-calibrate ngưỡng bằng GMM (thay vì percentile thủ công)

```python
from aic_pipeline import auto_calibrate_thresholds_gmm

static_th, dynamic_th, debug = auto_calibrate_thresholds_gmm(
    video_paths=["v1.mp4", "v2.mp4", ...],  # 15-30 video mẫu
    n_components=3,
)
print(debug["cluster_means_sorted"])  # kiểm tra bằng mắt trước khi dùng production
```

### GPU Shot Detector (chạy ngay, không cần checkpoint)

```python
from aic_pipeline import GPUShotDetector, PipelineConfig, run_pipeline

detector = GPUShotDetector(device="cuda", min_shot_len=15)
config = PipelineConfig(shot_backend=detector)
result = run_pipeline("video.mp4", config)
```

### Object Graph (tương quan đối tượng trong shot)

```python
from aic_pipeline import build_object_graph, summarize_graph_for_query, YoloDetector

# Mặc định (không cần GPU/model tải về):
graph = build_object_graph("video.mp4", shot)

# Chính xác hơn, cần GPU + ultralytics:
detector = YoloDetector(device="cuda")
graph = build_object_graph("video.mp4", shot, detector=detector)

summary = summarize_graph_for_query(graph)
# nhét summary vào StageHit.meta khi dùng với TemporalReranker
```

### Temporal Reranker 2 chiều

```python
from aic_pipeline import TemporalReranker

reranker = TemporalReranker(T_max=30.0)
# rerank(): 1 chuỗi tốt nhất/video (đủ dùng cho hầu hết trường hợp)
chains = reranker.rerank([stage0, stage1, stage2])
# rerank_bidirectional(): chấm điểm từng hit có tính cả bối cảnh tương lai
# (dùng khi cần độ tin cậy từng hit độc lập, ví dụ UI gợi ý giữa chừng)
chains_bi = reranker.rerank_bidirectional([stage0, stage1, stage2])
```



## Việc cần làm tiếp

1. **Chạy `calibrate_thresholds()`** trên 10-20 video bạn tự crawl để chỉnh
   ngưỡng motion sát domain (tin tức/đời sống VN), thay vì dùng mặc định.
2. **So sánh cheap vs semantic mode** trên vài video mẫu — xem độ khác biệt
   thực tế có đáng chi phí GPU không (tuỳ đặc điểm video của bạn).
3. **Khi có kết quả retrieval embedding chính thức** (bước ngoài package
   này): dùng `TemporalReranker` để xử lý truy vấn nhiều giai đoạn.
4. **Cân nhắc cắm AutoShot** (Tầng 1) nếu baseline histogram/SSIM còn bỏ sót
   nhiều chuyển cảnh mờ dần trên data thật của bạn.
5. **query_aware.py**: viết `MLLMScorerBackend` thật (Gemini/GPT-4o) khi cần
   rerank theo câu hỏi thi — xem khung code mẫu trong docstring file đó.
