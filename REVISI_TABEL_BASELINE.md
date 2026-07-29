# Tabel siap timpa untuk revisi pasca rerun baseline

Salinan **persis** tabel yang ada sekarang. Saat hasil rerun lengkap, ganti angkanya lalu
timpa tabel di berkas aslinya. Baris arsitektur usulan TIDAK berubah, jangan disentuh.

Sumber angka, tiap tar di `results/baselines/*_20260729_*` memuat
`<model>_recomputed_metrics.json` dan `train_efficiency.json`.

---

## LAPORAN — Tabel IV.10, klasifikasi 25 kelas

Berkas `docs/laporan-individu/bab-4/hasil_evaluasi.md`

**Yang berubah:** baris LOSVER, VulExplainer, LIVABLE + kolom n ketiganya

| Model                          | Akurasi           | Macro F1          | Weighted F1       | Banyak data test |
| ------------------------------ | ----------------- | ----------------- | ----------------- | ---------------- |
| Berbasis graph (usulan)        | 0,557 ± 0,009     | 0,548 ± 0,036     | 0,557 ± 0,008     | 913              |
| Hibrida graph–LM (usulan)      | 0,562 ± 0,028     | 0,529 ± 0,043     | 0,561 ± 0,026     | 913              |
| Sekuensial (usulan)            | 0,561 ± 0,013     | 0,553 ± 0,006     | 0,561 ± 0,012     | 913              |
| LOSVER (Nam & Baik, 2025)      | **0,631 ± 0,028** | **0,635 ± 0,030** | **0,631 ± 0,025** | 478              |
| VulExplainer (Fu et al., 2023) | 0,603 ± 0,014     | 0,593 ± 0,030     | 0,603 ± 0,014     | 915              |
| LIVABLE (Wen et al., 2023)     | 0,294 ± 0,033     | 0,041 ± 0,008     | 0,148 ± 0,011     | 533              |

---

## LAPORAN — Tabel IV.11, lokalisasi vuln-only

Berkas `docs/laporan-individu/bab-4/hasil_evaluasi.md`

**Yang berubah:** baris LOSVER + kolom n-nya

| Model                     | IFA ↓           | Top-1 ↑           | Top-5 ↑           | R@5%LOC ↑         | R@20%LOC ↑        | Effort@20%R ↓     | Banyak data test |
| ------------------------- | --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ---------------- |
| Berbasis graph (usulan)   | 12,12 ± 1,39    | 0,231 ± 0,007     | 0,548 ± 0,025     | 0,127 ± 0,014     | 0,368 ± 0,033     | 0,088 ± 0,017     | 590              |
| Hibrida graph–LM (usulan) | 13,59 ± 1,34    | 0,233 ± 0,012     | 0,535 ± 0,026     | 0,159 ± 0,016     | 0,377 ± 0,029     | 0,073 ± 0,012     | 590              |
| Sekuensial (usulan)       | 12,36 ± 1,15    | 0,243 ± 0,013     | 0,555 ± 0,024     | 0,133 ± 0,019     | 0,368 ± 0,030     | 0,089 ± 0,017     | 590              |
| LOSVER (Nam & Baik, 2025) | **3,72 ± 0,04** | **0,512 ± 0,010** | **0,770 ± 0,013** | **0,301 ± 0,002** | **0,581 ± 0,002** | **0,024 ± 0,002** | 476              |

---

## LAPORAN — Tabel IV.12, lokalisasi 26 kelas

Berkas `docs/laporan-individu/bab-4/hasil_evaluasi.md`

**Yang berubah:** baris LineVul dan LineVD + kolom n keduanya

| Model                                 | IFA ↓           | Top-1 ↑           | Top-5 ↑           | R@5%LOC ↑         | R@20%LOC ↑        | Effort@20%R ↓     | Banyak data test |
| ------------------------------------- | --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ---------------- |
| Berbasis graph (usulan)               | 11,72 ± 0,62    | 0,257 ± 0,003     | 0,583 ± 0,005     | 0,089 ± 0,015     | 0,289 ± 0,027     | 0,133 ± 0,016     | 584              |
| Hibrida graph–LM (usulan)             | 14,66 ± 0,51    | 0,245 ± 0,027     | 0,537 ± 0,022     | 0,097 ± 0,016     | 0,297 ± 0,036     | 0,119 ± 0,009     | 584              |
| Sekuensial (usulan)                   | 11,50 ± 0,91    | 0,273 ± 0,011     | 0,587 ± 0,021     | 0,105 ± 0,003     | 0,318 ± 0,005     | 0,110 ± 0,008     | 584              |
| LineVul (Fu & Tantithamthavorn, 2022) | 5,87 ± 0,12     | 0,221 ± 0,007     | 0,590 ± 0,008     | 0,138 ± 0,006     | 0,378 ± 0,004     | 0,085 ± 0,006     | 414              |
| LineVD (Hin et al., 2022)             | **0,40 ± 0,02** | **0,889 ± 0,005** | **0,973 ± 0,002** | **0,305 ± 0,005** | **0,523 ± 0,013** | **0,022 ± 0,001** | 599              |

---

## LAPORAN — Tabel D.3, per seed kelima baseline

Berkas `docs/laporan-individu/lampiran.md`

**Yang berubah:** SELURUH baris seed 1 dan seed 2, plus seed 42 untuk LOSVER, VulExplainer, LIVABLE

| Baseline | Seed | Macro F1 | Akurasi | Top-1 | Top-5 | IFA |
| --- | --- | --- | --- | --- | --- | --- |
| LOSVER | 42 | 0,658 | 0,638 | 0,523 | 0,756 | 3,76 |
| LOSVER | 1 | 0,645 | 0,655 | 0,508 | 0,782 | 3,72 |
| LOSVER | 2 | 0,601 | 0,600 | 0,504 | 0,771 | 3,67 |
| VulExplainer | 42 | 0,588 | 0,588 | — | — | — |
| VulExplainer | 1 | 0,626 | 0,615 | — | — | — |
| VulExplainer | 2 | 0,566 | 0,606 | — | — | — |
| LineVul | 42 | — | — | 0,214 | 0,582 | 6,01 |
| LineVul | 1 | — | — | 0,223 | 0,591 | 5,83 |
| LineVul | 2 | — | — | 0,228 | 0,597 | 5,77 |
| LineVD | 42 | — | — | 0,887 | 0,975 | 0,41 |
| LineVD | 1 | — | — | 0,895 | 0,972 | 0,38 |
| LineVD | 2 | — | — | 0,885 | 0,973 | 0,42 |
| LIVABLE | 42 | 0,042 | 0,259 | — | — | — |
| LIVABLE | 1 | 0,049 | 0,325 | — | — | — |
| LIVABLE | 2 | 0,033 | 0,299 | — | — | — |

---

## LAPORAN — Tabel H.2, efisiensi baseline

Berkas `docs/laporan-individu/lampiran.md`

**Yang berubah:** seluruh baris. Ambil dari train_efficiency.json gelombang 20260729 SAJA, jangan campur

| Baseline | Mean waktu total (jam) | Puncak VRAM | GPU |
| --- | --- | --- | --- |
| LOSVER | 3,27 | 6,1 GB | RTX 5090 |
| VulExplainer | 2,16 | 8,1 GB | RTX 5090 |
| LineVul | 0,30 | 12,9 GB | RTX 5090 |
| LineVD | 1,20 | 8,8 GB | RTX 4070 Ti Super |
| LIVABLE | 0,58 | 2,9 GB | RTX 4070 Ti Super |

---

## SLIDE — klasifikasi 25 kelas

Berkas `docs/laporan-individu/slides/05_hasil.md`

**Yang berubah:** sama dengan IV.10

| Model                          | Akurasi           | Macro F1          | Weighted F1       | n test |
| ------------------------------ | ----------------- | ----------------- | ----------------- | ------ |
| Berbasis graph (usulan)        | 0,557 ± 0,009     | 0,548 ± 0,036     | 0,557 ± 0,008     | 913    |
| Hibrida graph–LM (usulan)      | 0,562 ± 0,028     | 0,529 ± 0,043     | 0,561 ± 0,026     | 913    |
| Sekuensial (usulan)            | 0,561 ± 0,013     | 0,553 ± 0,006     | 0,561 ± 0,012     | 913    |
| **LOSVER** (Nam & Baik, 2025)  | **0,631 ± 0,028** | **0,635 ± 0,030** | **0,631 ± 0,025** | 478    |
| VulExplainer (Fu et al., 2023) | 0,603 ± 0,014     | 0,593 ± 0,030     | 0,603 ± 0,014     | 915 |
| LIVABLE (Wen et al., 2023)     | 0,294 ± 0,033     | 0,041 ± 0,008     | 0,148 ± 0,011     | 576    |

---

## SLIDE — lokalisasi vuln-only

Berkas `docs/laporan-individu/slides/05_hasil.md`

**Yang berubah:** sama dengan IV.11

| Model                     | IFA ↓           | Top-1 ↑           | Top-5 ↑           | R@5%LOC ↑         | R@20%LOC ↑        | Effort@20%R ↓     | n test |
| ------------------------- | --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------ |
| Berbasis graph (usulan)   | 12,12 ± 1,39    | 0,231 ± 0,007     | 0,548 ± 0,025     | 0,127 ± 0,014     | 0,368 ± 0,033     | 0,088 ± 0,017     | 590 |
| Hibrida graph–LM (usulan) | 13,59 ± 1,34    | 0,233 ± 0,012     | 0,535 ± 0,026     | 0,159 ± 0,016     | 0,377 ± 0,029     | 0,073 ± 0,012     | 590 |
| Sekuensial (usulan)       | 12,36 ± 1,15    | 0,243 ± 0,013     | 0,555 ± 0,024     | 0,133 ± 0,019     | 0,368 ± 0,030     | 0,089 ± 0,017     | 590 |
| **LOSVER**                | **3,72 ± 0,04** | **0,512 ± 0,010** | **0,770 ± 0,013** | **0,301 ± 0,002** | **0,581 ± 0,002** | **0,024 ± 0,002** | 476    |

---

## SLIDE — lokalisasi 26 kelas

Berkas `docs/laporan-individu/slides/05_hasil.md`

**Yang berubah:** sama dengan IV.12

| Model                     | IFA ↓           | Top-1 ↑           | Top-5 ↑           | R@5%LOC ↑         | R@20%LOC ↑        | Effort@20%R ↓     | n test |
| ------------------------- | --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ------ |
| Berbasis graph (usulan)   | 11,72 ± 0,62    | 0,257 ± 0,003     | 0,583 ± 0,005     | 0,089 ± 0,015     | 0,289 ± 0,027     | 0,133 ± 0,016     | 584 |
| Hibrida graph–LM (usulan) | 14,66 ± 0,51    | 0,245 ± 0,027     | 0,537 ± 0,022     | 0,097 ± 0,016     | 0,297 ± 0,036     | 0,119 ± 0,009     | 584 |
| Sekuensial (usulan)       | 11,50 ± 0,91    | 0,273 ± 0,011     | 0,587 ± 0,021     | 0,105 ± 0,003     | 0,318 ± 0,005     | 0,110 ± 0,008     | 584 |
| LineVul                   | 5,87 ± 0,12     | 0,221 ± 0,007     | 0,590 ± 0,008     | 0,138 ± 0,006     | 0,378 ± 0,004     | 0,085 ± 0,006     | 414 |
| **LineVD**                | **0,40 ± 0,02** | **0,889 ± 0,005** | **0,973 ± 0,002** | **0,305 ± 0,005** | **0,523 ± 0,013** | **0,022 ± 0,001** | 599    |

---

## SLIDE — Lampiran D.3 per seed

Berkas `docs/laporan-individu/slides/07_appendix.md`

**Yang berubah:** sama dengan Tabel D.3

| Baseline | Seed | Macro F1 | Akurasi | Top-1 | Top-5 | IFA |
| --- | --- | --- | --- | --- | --- | --- |
| LOSVER | 42 | 0,658 | 0,638 | 0,523 | 0,756 | 3,76 |
| LOSVER | 1 | 0,645 | 0,655 | 0,508 | 0,782 | 3,72 |
| LOSVER | 2 | 0,601 | 0,600 | 0,504 | 0,771 | 3,67 |
| VulExplainer | 42 | 0,588 | 0,588 | — | — | — |
| VulExplainer | 1 | 0,626 | 0,615 | — | — | — |
| VulExplainer | 2 | 0,566 | 0,606 | — | — | — |
| LineVul | 42 | — | — | 0,214 | 0,582 | 6,01 |
| LineVul | 1 | — | — | 0,223 | 0,591 | 5,83 |
| LineVul | 2 | — | — | 0,228 | 0,597 | 5,77 |
| LineVD | 42 | — | — | 0,887 | 0,975 | 0,41 |
| LineVD | 1 | — | — | 0,895 | 0,972 | 0,38 |
| LineVD | 2 | — | — | 0,885 | 0,973 | 0,42 |
| LIVABLE | 42 | 0,042 | 0,259 | — | — | — |
| LIVABLE | 1 | 0,049 | 0,325 | — | — | — |
| LIVABLE | 2 | 0,033 | 0,299 | — | — | — |

---

## SLIDE — Lampiran H efisiensi baseline

Berkas `docs/laporan-individu/slides/07_appendix.md`

**Yang berubah:** sama dengan Tabel H.2

| Baseline | Mean waktu total (jam) | Puncak VRAM | GPU |
| --- | --- | --- | --- |
| LOSVER | 3,27 | 6,1 GB | RTX 5090 |
| VulExplainer | 2,16 | 8,1 GB | RTX 5090 |
| LineVul | 0,30 | 12,9 GB | RTX 5090 |
| LineVD | 1,20 | 8,8 GB | RTX 4070 Ti Super |
| LIVABLE | 0,58 | 2,9 GB | RTX 4070 Ti Super |

---

## SLIDE — Cakupan Data Uji

Berkas `docs/laporan-individu/slides/07_appendix.md`

**Yang berubah:** baris lokalisasi LOSVER, LineVD, LineVul + klasifikasi

| Pengujian | Filter data uji | n |
|---|---|---|
| Klasifikasi 26 kelas | test penuh, benign dan rentan | 1.073 |
| Klasifikasi 25 kelas | test rentan | 913 |
| Lokalisasi usulan | baris penyebab jatuh ke node graph | 584 (26-cls), 590 (vuln-only) |
| Lokalisasi LineVD | baris penyebab jatuh ke node PDG | 599 |
| Lokalisasi LOSVER | berlabel, ≤512 token | 478 dinilai, 476 terhitung |
| Lokalisasi LineVul | ≤512 token | 414 |

---

## Cara mengisi angkanya

```bash
uv run python scripts/collect_baseline_results.py
```

Skrip itu membaca `results/baselines/*_20260729_*` di Drive, mengambil
`<model>_recomputed_metrics.json` dan `train_efficiency.json` terbaru per pasangan model dan
seed, lalu mencetak baris per seed beserta mean ± std dalam format tabel di atas, koma sebagai
desimal. Tinggal salin ke sel yang sesuai.

Dua penjaga di dalamnya.

- Pasangan model dan seed yang hasilnya belum ada dicetak sebagai `BELUM ADA`, tidak diisi
  dengan angka lama.
- Hanya LineVul dan LineVD **seed 42** yang boleh jatuh ke gelombang 20260707, karena keduanya
  sengaja tidak di-rerun. Model lain tidak akan pernah mengambil dari gelombang lama, supaya dua
  metodologi tidak tercampur tanpa terlihat.

Baris arsitektur usulan tidak ikut berubah dan tidak dicetak skrip ini. Salin apa adanya dari
tabel di atas.
