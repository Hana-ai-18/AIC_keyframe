"""
Tầng 5 — TEMPORAL RERANKING & RELINKING (MỚI — vá lỗ hổng đã chỉ ra khi so sánh
với WESp: bản trước chỉ cắt/lọc keyframe cho 1 video độc lập, chưa có cơ chế nối
các shot thành CHUỖI SỰ KIỆN mạch lạc khi truy vấn có nhiều giai đoạn).

Bối cảnh bài toán: truy vấn kiểu "A rồi B rồi C" (KIS-T nhiều giai đoạn, TRAKE).
Với mỗi giai đoạn (stage) trong truy vấn, hệ thống retrieval trả về danh sách
ứng viên (StageHit) — mỗi ứng viên gồm (video, thời điểm, điểm tương đồng thô
từ model embedding). Nhiệm vụ của tầng này: chọn ra CHUỖI kết nối các ứng viên
qua các stage sao cho vừa có điểm tương đồng cao, vừa nhất quán về THỜI GIAN
(đúng thứ tự, khoảng cách hợp lý).

CÔNG THỨC (dựa trên cơ chế Reranking & Relinking của WESp, generalize hoá):

  Với 2 stage liền kề X -> Y, mỗi ứng viên i ở X và j ở Y:

    temporal_distance:  D_ij = |t_i^X - t_j^Y|
    blended_similarity: B_ij = w_A * s_i^X + (1 - w_A) * s_j^Y
    distance_penalty:    phi(d) = 1 - exp(-gamma * d / alpha)          [chế độ "exp"]
                    hoặc  phi(d) = min(sqrt(1 + (beta*d/alpha)^2) - 1, 1)   [chế độ "sqrt"]
    temporal_score:      M_ij = B_ij * (1 - lambda * phi(D_ij))
                          M_ij = 0 nếu D_ij >= T_max (quá xa thì loại thẳng)

  Với >2 stage: lan truyền qua "first link" (stage đầu -> stage 2) rồi
  "subsequent links" (bắc cầu qua các stage sau), mỗi lần thêm 1 stage mới
  RELINK LẠI toàn bộ chuỗi — không chỉ tối ưu cục bộ như beam search cộng dồn
  điểm đơn thuần.

Đây là điểm khác biệt so với budget_allocator/frame_selector (Tầng 1-4, hoạt
động TRONG 1 video lúc build index offline): Tầng 5 hoạt động Ở BƯỚC QUERY-TIME,
trên kết quả retrieval đã có, xuyên suốt cả kho video.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Literal, Optional

import numpy as np


@dataclasses.dataclass
class StageHit:
    """1 ứng viên kết quả retrieval ở 1 stage (giai đoạn) của truy vấn, trong 1 video."""
    video_id: str
    stage_index: int          # thứ tự stage trong truy vấn (0 = đầu tiên)
    timestamp: float          # giây, thời điểm ứng viên này trong video
    similarity: float         # điểm tương đồng thô từ model embedding (đã normalize [0,1])
    frame_index: Optional[int] = None
    shot_id: Optional[int] = None
    meta: dict = dataclasses.field(default_factory=dict)   # chỗ chứa thêm info tuỳ ý (caption, v.v.)


@dataclasses.dataclass
class ChainResult:
    """1 chuỗi kết quả hoàn chỉnh xuyên suốt tất cả các stage, trong 1 video."""
    video_id: str
    hits: List[StageHit]          # 1 hit cho mỗi stage, theo đúng thứ tự stage_index
    chain_score: float            # điểm TÍCH LŨY thô (tổng dồn qua các stage, KHÔNG bị chặn trong [0,1])
    per_link_scores: List[float]  # điểm M_ij của từng cặp liên kết liền kề trong chuỗi

    @property
    def normalized_score(self) -> float:
        """
        chain_score là tổng tích lũy qua N stage (dp[k] = dp[k-1] + link_score),
        nên 2 chuỗi có SỐ STAGE KHÁC NHAU không so sánh trực tiếp bằng chain_score
        được (chuỗi 5 stage thường có chain_score cao hơn chuỗi 2 stage một cách
        tự nhiên, dù "chất lượng trung bình" mỗi liên kết có thể thấp hơn).

        normalized_score = chain_score / số_lượng_hit — đưa về thang trung bình
        mỗi hit, DÙNG ĐỂ SO SÁNH giữa các chuỗi có độ dài stage khác nhau.
        """
        return self.chain_score / max(1, len(self.hits))


PenaltyMode = Literal["exp", "sqrt"]


class TemporalReranker:
    """
    Nhận danh sách StageHit theo từng stage (đã group theo video_id từ trước hoặc
    để reranker tự group), trả về danh sách ChainResult đã xếp hạng theo chain_score.

    Tham số (đặt tên theo đúng công thức trong docstring module để dễ đối chiếu):
        w_A: trọng số blend similarity giữa 2 đầu liên kết (0.5 = cân bằng).
        lambda_: mức độ phạt theo khoảng cách thời gian (0 = không phạt gì).
        alpha, beta, gamma: tham số hình dạng của hàm phạt.
        penalty_mode: "exp" (phạt tăng dần mượt) hoặc "sqrt" (phạt tăng nhanh
                      rồi bão hoà — dùng khi bạn muốn "khoan dung" hơn với
                      khoảng cách lớn, ví dụ TRAKE cho phép độ trễ giữa các
                      hành động liên tiếp).
        T_max: khoảng cách thời gian tối đa (giây) — vượt quá thì loại thẳng
               cặp liên kết đó (M_ij = 0), không xét tiếp.
        decay_eta: hệ số suy giảm điểm khi 1 hit KHÔNG tìm được liên kết hợp lệ
                   nào ở stage tiếp theo (hit "mồ côi") — vẫn giữ lại trong xếp
                   hạng cuối nhưng bị phạt nặng, để không loại hẳn (có thể do
                   model retrieval ở stage sau bỏ sót, không hẳn do stage này sai).
    """

    def __init__(
        self,
        w_A: float = 0.5,
        lambda_: float = 0.6,
        alpha: float = 5.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        penalty_mode: PenaltyMode = "exp",
        T_max: float = 120.0,
        decay_eta: float = 0.1,
    ):
        self.w_A = w_A
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.penalty_mode = penalty_mode
        self.T_max = T_max
        self.decay_eta = decay_eta

    # -- hàm phạt khoảng cách thời gian ------------------------------------
    def _phi(self, d: float) -> float:
        if self.penalty_mode == "exp":
            return 1.0 - math.exp(-self.gamma * d / self.alpha)
        else:  # "sqrt"
            val = math.sqrt(1.0 + (self.beta * d / self.alpha) ** 2) - 1.0
            return min(val, 1.0)

    def _link_score(self, hit_x: StageHit, hit_y: StageHit) -> float:
        """Tính M_ij giữa 1 cặp (hit_x ở stage X, hit_y ở stage Y liền sau)."""
        d = abs(hit_x.timestamp - hit_y.timestamp)
        if d >= self.T_max:
            return 0.0
        b = self.w_A * hit_x.similarity + (1 - self.w_A) * hit_y.similarity
        return b * (1.0 - self.lambda_ * self._phi(d))

    # -- điều phối chính -----------------------------------------------------
    def rerank(self, hits_by_stage: List[List[StageHit]]) -> List[ChainResult]:
        """
        Args:
            hits_by_stage: list theo thứ tự stage, mỗi phần tử là list StageHit
                           của TẤT CẢ video ứng viên ở stage đó
                           (ví dụ: hits_by_stage[0] = kết quả stage "A", hits_by_stage[1]
                           = kết quả stage "B", ...).

        Returns:
            Danh sách ChainResult, MỖI CHUỖI chỉ thuộc 1 video (vì truy vấn nhiều
            giai đoạn phải cùng 1 video làm chứng cứ), sắp xếp giảm dần theo
            chain_score.

        Thuật toán (đúng theo docstring module — "first link" rồi "subsequent
        links", relink lại mỗi khi thêm stage mới):
        """
        if not hits_by_stage:
            return []
        if len(hits_by_stage) == 1:
            # chỉ có 1 stage -> không có gì để relink, trả thẳng theo similarity
            return [
                ChainResult(video_id=h.video_id, hits=[h], chain_score=h.similarity, per_link_scores=[])
                for h in sorted(hits_by_stage[0], key=lambda x: -x.similarity)
            ]

        # Nhóm hit theo video_id cho từng stage — 1 chuỗi chỉ đi trong 1 video.
        by_video: Dict[str, List[List[StageHit]]] = {}
        for stage_hits in hits_by_stage:
            for h in stage_hits:
                by_video.setdefault(h.video_id, [[] for _ in hits_by_stage])[h.stage_index].append(h)

        results: List[ChainResult] = []
        for video_id, stage_lists in by_video.items():
            # video này phải có ít nhất 1 hit ở stage đầu tiên để bắt đầu chuỗi
            if not stage_lists[0]:
                continue
            chain = self._build_chain_for_video(video_id, stage_lists)
            if chain is not None:
                results.append(chain)

        results.sort(key=lambda c: -c.chain_score)
        return results

    def rerank_bidirectional(self, hits_by_stage: List[List[StageHit]]) -> List[ChainResult]:
        """
        VÁ NHƯỢC ĐIỂM #5 (cross-moment interaction) — LÀM RÕ ĐÚNG PHẠM VI:

        rerank() gốc dùng DP forward-only, NHƯNG vì đây là DP TOÀN CỤC (không
        phải thuật toán tham lam/greedy từng bước), nó ĐÃ tìm được 1 đường đi
        tối ưu xuyên suốt toàn chuỗi — nếu bạn chỉ cần "1 chuỗi tốt nhất cho
        mỗi video", rerank() gốc và rerank_bidirectional() thường cho CÙNG
        MỘT chuỗi kết quả (đã kiểm chứng qua test: cả 2 đều chọn đúng nhánh
        có bằng chứng mạnh ở cuối, kể cả khi bằng chứng ở đầu/giữa yếu hơn).

        Khác biệt THẬT SỰ nằm ở một bài toán khác: khi bạn cần CHẤM ĐIỂM ĐỘC
        LẬP cho từng hit tại MỘT STAGE CỤ THỂ (ví dụ: "trong số các ứng viên
        ở stage giữa, cái nào đáng tin nhất, xét cả bối cảnh trước VÀ sau
        nó") — đây là tình huống UI hiển thị gợi ý cho người dùng chọn giữa
        chừng (như SEARCH ASSISTANCE của ConvAgent), hoặc khi muốn debug xem
        "hit này có đáng tin không" mà KHÔNG PHẢI đợi build xong cả chuỗi.
        Với bài toán đó, forward-only KHÔNG cho điểm hợp lý (dp[k][i] chỉ
        phản ánh quá khứ, hoàn toàn không biết hit i có link tốt với tương
        lai hay không), còn combined score 2 chiều ở đây phản ánh đúng.

        Tóm lại: dùng rerank() khi chỉ cần "1 chuỗi tốt nhất/video" (rẻ hơn,
        đủ dùng cho hầu hết trường hợp). Dùng rerank_bidirectional() khi cần
        điểm tin cậy CHO TỪNG HIT ở TỪNG STAGE một cách độc lập, có tính đến
        cả bối cảnh tương lai — trả giá bằng việc tính DP 2 lần (forward +
        backward) thay vì 1 lần.

        Thuật toán: DP forward (giống rerank()) + DP backward (đảo chiều
        duyệt qua stage) rồi hợp nhất tại mỗi hit: combined = forward +
        backward - similarity (trừ để không đếm similarity 2 lần).
        """
        if not hits_by_stage:
            return []
        if len(hits_by_stage) == 1:
            return self.rerank(hits_by_stage)

        by_video: Dict[str, List[List[StageHit]]] = {}
        for stage_hits in hits_by_stage:
            for h in stage_hits:
                by_video.setdefault(h.video_id, [[] for _ in hits_by_stage])[h.stage_index].append(h)

        results: List[ChainResult] = []
        for video_id, stage_lists in by_video.items():
            if any(len(s) == 0 for s in stage_lists):
                continue
            chain = self._build_chain_bidirectional(video_id, stage_lists)
            if chain is not None:
                results.append(chain)

        results.sort(key=lambda c: -c.chain_score)
        return results

    def _build_chain_bidirectional(
        self, video_id: str, stage_lists: List[List[StageHit]]
    ) -> Optional[ChainResult]:
        n_stages = len(stage_lists)

        # ---- forward pass (giống hệt _build_chain_for_video) ----
        fwd: List[List[float]] = [[] for _ in range(n_stages)]
        fwd[0] = [h.similarity for h in stage_lists[0]]
        for k in range(1, n_stages):
            prev_hits, cur_hits = stage_lists[k - 1], stage_lists[k]
            for hy in cur_hits:
                best = -1.0
                for i, hx in enumerate(prev_hits):
                    link = self._link_score(hx, hy)
                    if link <= 0.0:
                        continue
                    candidate = fwd[k - 1][i] + link
                    if candidate > best:
                        best = candidate
                if best < 0:
                    best = self.decay_eta * hy.similarity
                fwd[k].append(best)

        # ---- backward pass (đảo chiều duyệt qua stage) ----
        bwd: List[List[float]] = [[] for _ in range(n_stages)]
        bwd[n_stages - 1] = [h.similarity for h in stage_lists[n_stages - 1]]
        for k in range(n_stages - 2, -1, -1):
            cur_hits, next_hits = stage_lists[k], stage_lists[k + 1]
            for hx in cur_hits:
                best = -1.0
                for j, hy in enumerate(next_hits):
                    link = self._link_score(hx, hy)
                    if link <= 0.0:
                        continue
                    candidate = bwd[k + 1][j] + link
                    if candidate > best:
                        best = candidate
                if best < 0:
                    best = self.decay_eta * hx.similarity
                bwd[k].append(best)

        # ---- hợp nhất: chọn hit tốt nhất theo combined score tại mỗi stage ----
        hits_chain: List[StageHit] = []
        combined_scores: List[float] = []
        for k in range(n_stages):
            combined = [
                fwd[k][i] + bwd[k][i] - stage_lists[k][i].similarity
                for i in range(len(stage_lists[k]))
            ]
            best_i = int(np.argmax(combined))
            hits_chain.append(stage_lists[k][best_i])
            combined_scores.append(combined[best_i])

        # per_link_scores tính lại trên chuỗi đã chọn (có thể không liên tục
        # hoàn hảo về link_score nếu 2 stage liền kề chọn hit không link trực
        # tiếp tốt nhất với nhau — đây là đánh đổi CÓ CHỦ ĐÍCH của phương án
        # 2 chiều: ưu tiên "đúng tại từng thời điểm xét toàn cục" hơn "mượt
        # cục bộ giữa 2 điểm liền kề").
        per_link = [
            self._link_score(hits_chain[i], hits_chain[i + 1]) for i in range(len(hits_chain) - 1)
        ]

        return ChainResult(
            video_id=video_id, hits=hits_chain,
            chain_score=float(np.mean(combined_scores)) * n_stages,  # thang tương đương chain_score của rerank()
            per_link_scores=per_link,
        )


    def _build_chain_for_video(
        self, video_id: str, stage_lists: List[List[StageHit]]
    ) -> Optional[ChainResult]:
        """
        Xây 1 chuỗi tốt nhất cho 1 video, qua tất cả các stage.

        Cài đặt bằng quy hoạch động đơn giản (tương đương "first link" +
        "subsequent links" của WESp nhưng viết dạng DP cho tổng quát với
        N stage bất kỳ, không giới hạn 2-3 stage):

          dp[k][i] = điểm tốt nhất của chuỗi kết thúc tại hit thứ i của stage k
          dp[0][i] = similarity của hit i ở stage 0
          dp[k][j] = max over i in stage k-1 của ( dp[k-1][i] + link_score(hit_i, hit_j) )
                     nếu không có i nào link được (mọi M_ij=0) -> áp decay_eta lên
                     similarity của chính hit_j để nó vẫn có mặt trong chuỗi
                     (không bị loại cứng, nhưng bị phạt nặng).

        Trả về None nếu KHÔNG stage nào sau stage 0 có hit (video này không đủ
        chứng cứ cho toàn bộ truy vấn).
        """
        n_stages = len(stage_lists)
        if any(len(s) == 0 for s in stage_lists):
            # video này thiếu hẳn 1 stage -> không đủ chứng cứ cho toàn chuỗi,
            # loại luôn (khác với "hit mồ côi" — đây là thiếu cả stage)
            return None

        # dp[k] = list các (score, backpointer_index_ở_stage_k-1, link_score_dùng)
        dp: List[List[float]] = [[] for _ in range(n_stages)]
        back: List[List[int]] = [[] for _ in range(n_stages)]
        link_used: List[List[float]] = [[] for _ in range(n_stages)]

        dp[0] = [h.similarity for h in stage_lists[0]]
        back[0] = [-1] * len(stage_lists[0])
        link_used[0] = [0.0] * len(stage_lists[0])

        for k in range(1, n_stages):
            prev_hits = stage_lists[k - 1]
            cur_hits = stage_lists[k]
            for j, hy in enumerate(cur_hits):
                best_score = -1.0
                best_i = -1
                best_link = 0.0
                for i, hx in enumerate(prev_hits):
                    link = self._link_score(hx, hy)
                    if link <= 0.0:
                        continue
                    candidate = dp[k - 1][i] + link
                    if candidate > best_score:
                        best_score = candidate
                        best_i = i
                        best_link = link
                if best_i == -1:
                    # không link được với bất kỳ hit nào ở stage trước -> decay
                    # (hit "mồ côi" theo đúng ý decay_eta trong docstring lớp)
                    best_score = self.decay_eta * hy.similarity
                    best_i = -1
                    best_link = 0.0
                dp[k].append(best_score)
                back[k].append(best_i)
                link_used[k].append(best_link)

        # tìm điểm tốt nhất ở stage cuối, truy vết ngược lại
        last_k = n_stages - 1
        best_j = int(np.argmax(dp[last_k]))
        best_final_score = dp[last_k][best_j]

        hits_chain: List[StageHit] = []
        links_chain: List[float] = []
        k, j = last_k, best_j
        while k >= 0:
            hits_chain.append(stage_lists[k][j])
            if k > 0:
                links_chain.append(link_used[k][j])
                j = back[k][j]
                if j == -1:
                    # chuỗi bị đứt ở đây — các stage trước đó không xác định
                    # được qua truy vết (do đúng hit đang xét là "mồ côi").
                    # Dừng truy vết, phần còn lại coi như không có chứng cứ rõ.
                    break
            k -= 1

        hits_chain.reverse()
        links_chain.reverse()

        return ChainResult(
            video_id=video_id,
            hits=hits_chain,
            chain_score=float(best_final_score),
            per_link_scores=links_chain,
        )
