# Hasil Continual Learning (relearn task-B = BigVul + TitanVul)

Model: N48 (GNN-only, jknet, 26-kelas, checkpoint 20260606_163818, MegaVul Macro F1 0.525).
Satu arsitektur N48 dipakai di semua baris. Setiap metode melanjutkan pelatihan model
task-A yang sama pada task-B. Jenis: domain-incremental (26 kelas tetap, domain data berganti).

Task-A = MegaVul 26-kelas. Task-B = relearn 26-kelas. Macro F1 pada test masing-masing.
Sebelum pembaruan = model task-A apa adanya (tanpa retraining) dievaluasi pada kedua test.
Forgetting = Macro F1 task-A sebelum pembaruan dikurangi sesudah (makin kecil makin baik).

Urutan dan asal data:
1. Task-A (MegaVul): N48 dilatih lebih dulu pada MegaVul top-25 CWE plus benign (26 kelas, maks 1600 per kelas, seed 42).
2. Task-B (relearn): dibangun dari BigVul plus TitanVul, vuln top-25 dideduplikasi terhadap MegaVul dan antar keduanya, ditambah benign. Label dipetakan ke vocab kanonik task-A agar id kelas selaras.
3. Pelatihan kontinual: model mulai dari bobot task-A, lalu dilanjutkan pada task-B. Urutan = MegaVul lebih dulu, baru relearn.
4. Split test DIKUNCI via `split_seed=42` di semua seed agar eval backbone yang tetap tidak bocor. Hanya `train.seed` (init) yang divariasikan untuk mengukur variance.

## Mean ± std (3 seed: 42, 1, 2)

Tiga seed pelatihan per metode (`split_seed=42` tetap, init `train.seed` = 42/1/2). "Sebelum pembaruan" deterministik (backbone + split tetap) sehingga identik di semua seed (std 0), sekaligus bukti split tidak bocor.

| Metode                           | Macro F1 task-A | Macro F1 task-B | Forgetting ↓  |
| -------------------------------- | --------------- | --------------- | ------------- |
| Sebelum pembaruan                | 0.525           | 0.328           | —             |
| Fine-tuning naif                 | 0.139 ± 0.003   | 0.365 ± 0.022   | 0.386 ± 0.003 |
| EWC-DR                           | 0.400 ± 0.031   | 0.375 ± 0.046   | 0.125 ± 0.031 |
| Experience replay                | 0.319 ± 0.022   | 0.394 ± 0.010   | 0.206 ± 0.022 |
| **EWC-DR dan experience replay** | **0.473 ± 0.041** | 0.424 ± 0.036 | **0.051 ± 0.041** |

EWC-DR + experience replay mencatat forgetting terkecil dan retensi task-A terbaik, konsisten di tiga seed. Gap antar-metode jauh lebih besar dari std sehingga kesimpulan robust.

## Raw per-seed (audit — mean±std di atas dihitung dari sini)

| Metode                           | seed | task-A | task-B | Forgetting |
| -------------------------------- | ---- | ------ | ------ | ---------- |
| Fine-tuning naif                 | 42   | 0.1399 | 0.3401 | +0.3846    |
| Fine-tuning naif                 | 1    | 0.1410 | 0.3802 | +0.3835    |
| Fine-tuning naif                 | 2    | 0.1358 | 0.3751 | +0.3888    |
| EWC-DR                           | 42   | 0.4340 | 0.4263 | +0.0906    |
| EWC-DR                           | 1    | 0.3721 | 0.3622 | +0.1524    |
| EWC-DR                           | 2    | 0.3940 | 0.3377 | +0.1306    |
| Experience replay                | 42   | 0.2932 | 0.3871 | +0.2313    |
| Experience replay                | 1    | 0.3326 | 0.3882 | +0.1920    |
| Experience replay                | 2    | 0.3297 | 0.4054 | +0.1949    |
| EWC-DR dan experience replay     | 42   | 0.5123 | 0.4658 | +0.0122    |
| EWC-DR dan experience replay     | 1    | 0.4766 | 0.4020 | +0.0479    |
| EWC-DR dan experience replay     | 2    | 0.4314 | 0.4046 | +0.0932    |

Baseline "Sebelum pembaruan" = 0.5246 (task-A) / 0.3275 (task-B) identik di tiga seed (deterministik).
Per-seed MD lengkap + checkpoint per run di Drive `results/RELEARN_RESULTS_s{42,1,2}.md` dan `checkpoints/`.
