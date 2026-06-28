# Hasil Continual Learning Class-Incremental (CIL, task-B = 10 CWE baru MegaVul)

Model: N48 (GNN-only, jknet, checkpoint task-A 20260606_163818, MegaVul Macro F1 0.525).
Satu arsitektur N48; head diperluas 26->36 (load expandable) untuk menampung 10 kelas baru.
Jenis: class-incremental (kelas BARU ditambah, bukan domain). Sesuai setting utama paper EWC-DR.

Task-A = MegaVul 26 kelas lama. Task-B = 10 CWE baru (megavul_cil, id 26..35).
Sebelum pembaruan task-B = N.A. (model 26 kelas belum punya head untuk kelas baru).
Forgetting = Macro F1 task-A sebelum pembaruan dikurangi sesudah (makin kecil makin baik).
A_last = metrik pada SEMUA kelas terlihat (gabungan task-A + task-B, 36 kelas) setelah
task terakhir; A_avg = rata-rata akurasi setelah tiap task (A_1 task-A, A_2 = A_last).
A_last dan A_avg mengikuti protokol CIL paper EWC-DR; akurasi seperti paper, plus macro-F1.

Urutan dan asal data:
1. Task-A (MegaVul): N48 dilatih pada MegaVul top-25 CWE plus benign (26 kelas).
2. Task-B (megavul_cil): 10 CWE non-top25 paling banyak di MegaVul (CWE-119,190,362,264,399,400,401,189,617,835), label dipetakan ke 26..35.
3. Pelatihan kontinual: mulai dari bobot task-A (head 26->36), lanjut pada task-B.
4. Split test DIKUNCI via `split_seed=42` di semua seed agar eval backbone tetap tidak bocor. Hanya `train.seed` (init) divariasikan.

## Mean ± std (3 seed: 42, 1, 2)

Tiga seed pelatihan per metode (`split_seed=42` tetap, init `train.seed` = 42/1/2). "Sebelum pembaruan" task-A deterministik (0.5246, std 0 di semua seed) sekaligus bukti split tidak bocor.

| Metode                           | F1 task-A     | F1 task-B     | A_last F1 (36) | A_last Acc (36) | A_avg Acc     | Forgetting ↓  |
| -------------------------------- | ------------- | ------------- | -------------- | --------------- | ------------- | ------------- |
| Sebelum pembaruan                | 0.525         | —             | —              | —               | —             | —             |
| Fine-tuning naif                 | 0.000 ± 0.000 | 0.605 ± 0.016 | 0.090 ± 0.003  | 0.214 ± 0.002   | 0.361 ± 0.001 | 0.525 ± 0.000 |
| EWC-DR                           | 0.097 ± 0.065 | 0.546 ± 0.016 | 0.149 ± 0.044  | 0.213 ± 0.017   | 0.360 ± 0.008 | 0.428 ± 0.065 |
| Experience replay                | 0.357 ± 0.022 | 0.519 ± 0.022 | 0.336 ± 0.012  | 0.308 ± 0.014   | 0.407 ± 0.007 | 0.168 ± 0.022 |
| **EWC-DR dan experience replay** | **0.495 ± 0.015** | 0.323 ± 0.069 | **0.385 ± 0.015** | **0.361 ± 0.021** | **0.434 ± 0.010** | **0.030 ± 0.015** |

EWC-DR + experience replay memimpin A_last F1, A_last Acc, A_avg, dan forgetting terkecil, konsisten di tiga seed. Gap antar-metode jauh di atas std.

## Raw per-seed (audit — mean±std di atas dihitung dari sini)

| Metode                       | seed | F1-A   | F1-B   | A_last F1 | A_last Acc | A_avg Acc | Forget  |
| ---------------------------- | ---- | ------ | ------ | --------- | ---------- | --------- | ------- |
| Fine-tuning naif             | 42   | 0.0000 | 0.6229 | 0.0901    | 0.2158     | 0.3614    | +0.5246 |
| Fine-tuning naif             | 1    | 0.0000 | 0.5914 | 0.0875    | 0.2114     | 0.3592    | +0.5246 |
| Fine-tuning naif             | 2    | 0.0000 | 0.6002 | 0.0938    | 0.2152     | 0.3611    | +0.5246 |
| EWC-DR                       | 42   | 0.1685 | 0.5324 | 0.1970    | 0.2313     | 0.3692    | +0.3561 |
| EWC-DR                       | 1    | 0.0425 | 0.5412 | 0.1104    | 0.1978     | 0.3524    | +0.4821 |
| EWC-DR                       | 2    | 0.0795 | 0.5642 | 0.1403    | 0.2108     | 0.3589    | +0.4451 |
| Experience replay            | 42   | 0.3644 | 0.5361 | 0.3428    | 0.3128     | 0.4099    | +0.1601 |
| Experience replay            | 1    | 0.3321 | 0.5267 | 0.3230    | 0.3184     | 0.4127    | +0.1924 |
| Experience replay            | 2    | 0.3744 | 0.4946 | 0.3432    | 0.2923     | 0.3996    | +0.1502 |
| EWC-DR dan experience replay | 42   | 0.4905 | 0.2591 | 0.3701    | 0.3445     | 0.4258    | +0.0341 |
| EWC-DR dan experience replay | 1    | 0.4822 | 0.3960 | 0.4000    | 0.3843     | 0.4457    | +0.0424 |
| EWC-DR dan experience replay | 2    | 0.5117 | 0.3141 | 0.3836    | 0.3545     | 0.4307    | +0.0129 |

Baseline "Sebelum pembaruan" task-A = 0.5246 identik di tiga seed (deterministik).
Per-seed MD lengkap + checkpoint per run di Drive `results/RELEARN_CIL_RESULTS_s{42,1,2}.md` dan `checkpoints/`.
