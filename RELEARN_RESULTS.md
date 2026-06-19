# Hasil Continual Learning (relearn task-B = BigVul + TitanVul)

Model: N48 (GNN-only, jknet, 26-kelas, checkpoint 20260606_163818, MegaVul Macro F1 0.525).
Satu arsitektur N48 dipakai di semua baris. Setiap metode melanjutkan pelatihan model
task-A yang sama pada task-B. Jenis: domain-incremental (26 kelas tetap, domain data berganti).

Task-A = MegaVul 26-kelas. Task-B = relearn 26-kelas. Macro F1 pada test masing-masing.
Sebelum pembaruan = model task-A apa adanya (tanpa retraining) dievaluasi pada kedua test.
Forgetting = Macro F1 task-A sebelum pembaruan dikurangi sesudah (makin kecil makin baik).

| Metode | Macro F1 task-A | Macro F1 task-B | Forgetting ↓ |
|---|---|---|---|
| Sebelum pembaruan | 0.525 | 0.281 | — |
| Fine-tuning naif | 0.115 | 0.305 | +0.409 |
| EWC-DR | 0.291 | 0.360 | +0.234 |
| Experience replay | 0.340 | 0.339 | +0.184 |
| EWC-DR dan experience replay | 0.385 | 0.290 | +0.140 |
