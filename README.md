# CWFC — Compression as Conditional Optimal Transport (de-risk code)

Bốn script de-risk đã test, chép verbatim từ Phụ lục A1–A4 của `brainstorm/idea.md`.
Mọi con số ở Phần 1 và Phần 3 của tài liệu được sinh ra từ đúng các bản này —
**không chỉnh sửa** khi chạy lại local.

## Nội dung folder

| file | vai trò | phụ thuộc ngoài | trạng thái |
|---|---|---|---|
| `derisk_cot.py` | **Cơ chế lõi** (Phần 3, E1–E6): extended-cost conditional Sinkhorn. Q1 = hình học coupling, Q2 = frontier huấn luyện, hook `--cy` (raw/projD/poolD). | chỉ `torch`, `numpy` | test đầy đủ |
| `derisk_mixture_entropy.py` | De-risk OT-entropy (Phần 1.1, **đã KILL**). Mixture K-thành-phần vs single-Gaussian, coder rANS thật (lossless verified). Giữ làm tài liệu âm + hạ tầng coder. | `compressai`, `torchvision`, `pillow` | tested |
| `derisk_dp.py` | Traversal A/B/C trên mặt phẳng PSNR–LPIPS (Phần 1.2). Bài học C-path (round-trip VAE phá traverse). | `vae`: `diffusers`; `lpips`: `lpips` (chế độ `toy`/`feat` không cần) | logic tested; SD-VAE path đã chạy |
| `wflow_endpoint.py` | Gate G3: inversion W-Flow về nhất quán ô lượng tử. `invert_to_consistency` ĐÃ TEST; `WFlowAdapter` là skeleton `# WIRE` chưa test. | `torch` (self-test); W-Flow checkout để wire adapter | inversion tested (~9500× residual↓) |

## Cài đặt

```bash
pip install -r requirements.txt
```

Tối thiểu để chạy phần cơ chế (`derisk_cot.py`) chỉ cần `torch` + `numpy`.

## Lệnh tái lập kết quả chính (Phần 4.3)

```bash
# Cơ chế núm ở thấp chiều (dy=2): hai đầu mút khớp anchor lý thuyết
python derisk_cot.py --q both --N 1024 --steps 4000

# Vách đá cao chiều (dy=64, c_y thô)
python derisk_cot.py --q 2 --dy_q2 64 --cy raw   --N 1024 --steps 4000

# Cách chữa: embedding thấp chiều mở lại vách đá thành gradient
python derisk_cot.py --q 2 --dy_q2 64 --cy proj4 --N 1024 --steps 4000

# Hình học ở cỡ latent nén thật (dy=4096), không cần train
python -c "from derisk_cot import q1_coupling; \
  q1_coupling(dy_list=(4096,), N=1024, cy_kind='raw',   device='cuda'); \
  q1_coupling(dy_list=(4096,), N=1024, cy_kind='proj8', device='cuda')"

# De-risk OT-entropy (đã kill) — cần một folder ảnh
python derisk_mixture_entropy.py --data /path/to/images --ks 1 3 5

# Traversal D-P: sandbox (không download) rồi bản quyết định (SD-VAE + LPIPS)
python derisk_dp.py --data /path/to/images --backbone toy --percep feat
python derisk_dp.py --data /path/to/images --backbone vae --percep lpips --delta 0.7 --n_t 9 --max_images 200

# Self-test inversion (chỉ torch, CPU)
python wflow_endpoint.py
```

## Ghi chú quan trọng

- `η*` (dải chuyển tiếp của núm) **phụ thuộc N và ε**: đổi cấu hình phải chạy lại
  Q2 lưới thưa để map lại dải. Bản thật nên chuẩn hoá η theo quantile của c_y/batch.
- SEED cố định giữa các η để so công bằng; bảng công bố cần trung bình ≥2–3 seed
  (nhiễu run-đơn ~0.1 distortion).
- Toàn bộ bằng chứng hiện tại là toy 2D — chỉ **cơ chế, thứ tự, dạng đường cong**
  chuyển sang ảnh thật, không phải số tuyệt đối. Bước kế: G1 pilot cỡ ảnh nhỏ.

Xem `brainstorm/idea.md` cho lý thuyết đầy đủ, hồ sơ thực nghiệm E1–E6, và roadmap G1–G4.

---

## G1 pilot — `g1_pilot.py` (cầu nối toy → W-Flow)

Nâng đúng mục tiêu extended-cost từ toy 2D lên **ảnh nhỏ**, với điều kiện ŷ là
**latent lượng tử của một AE tí hon** (có cấu trúc không gian như code truyền thật).
Trả lời ba câu hỏi PASS/KILL của G1 (Phần 5):

1. Vai trò kép của η + dải núm có tái hiện khi ŷ là latent 3D không?
2. Embedding ngữ nghĩa (`pool`) có ≥ random-proj (`projD`) và raw không?
3. Chốt metric realism ở cỡ ảnh: **MMD trên random-feature cố định** + **diversity
   `Var_z[G(z,ŷ)]`** (bắt mode-collapse).

**Thành phần:** AE conv lượng tử (train trước, đóng băng) → ŷ; generator conv nhỏ
`G(z,ŷ)`; loss extended-cost. Hai bản loss:
- `--loss crude` — đúng công thức A1 (plan-weighted, P detach). Ở ảnh, squared-cost
  làm generator hồi quy về trung bình có điều kiện → collapse (đối chứng).
- `--loss debiased` (mặc định) — thêm số hạng **gen-gen của Sinkhorn divergence**
  (bản "barycentric debiased" mà Mục 2.5/3.7 chỉ tới) để thưởng độ trải mẫu sinh,
  khôi phục đầu mút perception.

**Metric:** PSNR (fidelity, ↑) · MMD-RFF (realism, ↓) · `Var_z` (diversity: cao ở
η thấp, →0 khi η→∞; ~0 ở MỌI η = collapse).

### Chạy trên Kaggle (GPU)

CIFAR-10 tự tải qua torchvision (bật Internet trong Kaggle, hoặc add dataset và trỏ
`root`). Cell notebook sẵn: xem `kaggle_g1.py`.

```bash
# Bản chính: CIFAR, 3 embedding, quét η, 2–3 seed, lưu JSON
python g1_pilot.py --dataset cifar --H 32 --mode both \
    --N 1024 --steps 4000 --ae_steps 3000 \
    --seeds 0 1 2 --etas 0 0.1 0.3 1 3 10 100 \
    --embeds raw proj8 pool --loss debiased --out g1_cifar.json

# Đối chứng loss thô (kỳ vọng collapse ở ảnh) — chỉ để so
python g1_pilot.py --dataset cifar --H 32 --N 1024 --steps 4000 \
    --embeds pool --loss crude --out g1_crude.json

# Smoke nhanh (synth, không download) để kiểm plumbing trước
python g1_pilot.py --smoke
```

**Đọc PASS/KILL:** với ≥1 embedding, PSNR đơn điệu tăng theo η *và* MMD giảm-rồi-tăng
*và* `Var_z` cao→0 ⇒ frontier D-P tái hiện trên ảnh ⇒ PASS, sang G2. `pool ≥ proj`
⇒ chốt c_y ngữ nghĩa. Nếu KHÔNG embedding nào có dải trung gian (chỉ nhảy hai đầu
mút) ⇒ KILL/RETHINK: chuyển blocked-OT / per-sample marginal (Phần 5).

### Phát hiện đã có (CPU, budget vừa) + giới hạn cần biết trước khi chạy Kaggle

Trên synth (N=256, 1500 steps, `pool`, debiased):

| η | PSNR | MMD | Div(Var_z) |
|---|---|---|---|
| 0 | 14.0 | 9.3e-3 | 2e-7 |
| 0.3 | 27.6 | 9.6e-4 | 1e-6 |
| 1–30 | 28.8–29.0 | ~9.4e-4 | ~1e-6 |

- **Trục fidelity + núm η CHẠY**: PSNR đơn điệu 14→29, MMD cải thiện mạnh theo η,
  có dải dùng được (η≥0.3). Câu hỏi (1) của G1 có tín hiệu dương.
- **Trục diversity COLLAPSE ở mọi η** (`Var_z~1e-6`), kể cả η=0, và **không phải
  do budget nhỏ** — nó có tính cấu trúc: loss OT squared-cost + entropic có thiên
  lệch barycentric kéo về trung bình có điều kiện; ở gần-collapse số hạng debias
  có gradient≈0 nên không bơm được độ trải. Đầu mút perception (mẫu đa dạng) **KHÔNG
  nảy sinh từ loss plan-detach** (thô hay debiased) — cần bộ máy generative-OT thật
  của W-Flow (potential không-detach + eps-annealing / gen-to-gen thật) = phần G4.

**Knob để khám phá trục diversity trên Kaggle full-budget** (theo thứ tự đáng thử):
`--eps 0.01` (plan sắc hơn, bớt mờ barycentric) · `--debias_w 1.0..2.0` (tăng lực
chống-collapse) · so `--loss crude` vs `debiased` để thấy tác dụng debias · N và
steps lớn (1024 / ≥4000) như toy E3.

**Kết luận pilot G1 (tạm):** núm + fidelity + so-embedding tái hiện được trên ảnh
(PASS phần đó); trục perception/diversity khoanh đúng chỗ cần G4. Nếu ở Kaggle với
mọi knob trên mà `Var_z` vẫn collapse → xác nhận loss plan-detach không đủ, chuyển
sang Sinkhorn-divergence potential-based (geomloss-style) — là hạng mục đã định vị.
