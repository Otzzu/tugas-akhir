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
4. Split test seed 42 (80/10/10); importance EWC-DR dan buffer replay dari train task-A.

| Metode | F1 task-A | F1 task-B | A_last F1 (36) | A_last Acc (36) | A_avg Acc | Forgetting ↓ |
|---|---|---|---|---|---|---|
| Sebelum pembaruan | 0.5246 | — | — | — | — | — |
| Fine-tuning naif | 0.0000 | 0.6018 | 0.0899 | 0.2102 | 0.3586 | +0.5246 |
| EWC-DR | 0.0443 | 0.5533 | 0.1115 | 0.2002 | 0.3536 | +0.4803 |
| Experience replay | 0.3745 | 0.5366 | 0.3554 | 0.3109 | 0.4090 | +0.1501 |
| EWC-DR dan experience replay | 0.4748 | 0.3274 | 0.3753 | 0.3632 | 0.4351 | +0.0498 |

Checkpoint terlatih (Drive checkpoints/, untuk re-evaluasi tanpa latih ulang):
- Fine-tuning naif: `20260619_213909_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/cil/)
- EWC-DR: `20260619_215528_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/cil/)
- Experience replay: `20260619_220817_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/cil/)
- EWC-DR dan experience replay: `20260619_222906_lmgat_codebert_multiclass_checkpoints.zip` (config: lihat configs/ablation/relearn/cil/)
