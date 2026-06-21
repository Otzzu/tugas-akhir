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
4. Split test seed 42 (80/10/10); importance EWC-DR dan buffer replay diambil dari split train task-A tanpa kebocoran ke test.

| Metode | Macro F1 task-A | Macro F1 task-B | Forgetting ↓ |
|---|---|---|---|
| Sebelum pembaruan | 0.525 | 0.281 | — |
| Fine-tuning naif | 0.158 | 0.304 | +0.366 |
| EWC-DR | 0.416 | 0.426 | +0.109 |
| Experience replay | 0.450 | 0.381 | +0.075 |
| EWC-DR dan experience replay | 0.438 | 0.357 | +0.087 |

Checkpoint terlatih (Drive checkpoints/, untuk re-evaluasi tanpa latih ulang):
- Fine-tuning naif: `20260619_200234_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/)
- EWC-DR: `20260619_200708_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/)
- Experience replay: `20260619_201757_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/)
- EWC-DR dan experience replay: `20260619_202816_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/)
