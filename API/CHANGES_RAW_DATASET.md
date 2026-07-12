# Perubahan API — Penyimpanan Data Raw Dataset

**Tanggal perubahan: 12 Juli 2026** — satu gelombang rilis bersama `gnn-vuln 0.1.12`;
berlaku untuk ketiga bagian dokumen ini (raw dataset §1–5, role datasets §6–7,
multi-task continual §8).

Tujuan: data raw yang dipakai membangun tiap dataset `.pt` tersimpan permanen, bisa
diunduh, dan lineage raw → .pt → model bisa ditelusuri dari database API.

## 1. Raw rows disimpan permanen di object storage (baru)

Setiap ingest yang sukses mengunggah baris raw ke MinIO (bucket `datasets`) sebagai
`{raw_id}.json` — **JSON array dengan bentuk yang sama persis dengan format upload**
(DatasetRow), termasuk `id` dan `cve_id`. Row tanpa `id` mendapat id hasil generate
(posisi row) yang sudah dimaterialisasi, jadi file raw langsung bisa di-upload ulang.

Isi raw TIDAK disimpan di database — database hanya menyimpan metadata + pointer.

## 2. Tabel baru `raw_datasets` + relasi dari `datasets` (perubahan schema DB)

Raw jadi entitas kelas satu dengan tabelnya sendiri, karena **satu raw bisa dipakai
membangun banyak dataset .pt** (rows sama, config build berbeda).

| Tabel/kolom | Isi |
|---|---|
| `raw_datasets.id` PK | `raw_` + sha256(konten)[:12] — content-addressed: upload konten identik dedup ke baris + objek yang sama |
| `raw_datasets.storage_uri` | Pointer S3 ke `{raw_id}.json` |
| `raw_datasets.num_rows`, `size_bytes`, `content_hash` | Metadata integritas |
| `datasets.raw_id` FK → raw_datasets | Relasi dataset → raw (N:1) |
| `datasets.source_dataset_ids` JSON | Untuk dataset hasil merge: dataset sumbernya |

Lineage di level database:

```
models.dataset_id           -> datasets
datasets.raw_id             -> raw_datasets -> storage_uri -> file raw JSON di S3
datasets.source_dataset_ids -> dataset sumber (merge), masing-masing punya raw_id
datasets.data_config_id     -> configs (cara .pt dibangun, immutable)
relearn_jobs.dataset_ids + result_model_id -> jejak training -> model
models.base_model_id        -> rantai continual antar model
```

Migrasi: belum ada deployment — DB baru otomatis dapat tabel+kolom via `create_all`
saat boot. DB dev lama: recreate (`docker compose down -v`).

## 3. Endpoint baru: unduh raw

```
GET /datasets/{dataset_id}/raw
```

Mengembalikan JSON array baris raw persis yang dipakai membangun `.pt` dataset itu.

Respons job ingest (`POST /datasets`, `POST /datasets/upload`, `GET /datasets/jobs/*`)
kini menyertakan `raw_id` saat status done, jadi job → raw tertaut langsung tanpa
lookup tambahan.

- Dataset hasil **merge** tidak punya raw sendiri → 404 dengan petunjuk
  `source_dataset_ids` (ambil raw dari tiap sumber).
- Dataset lama (sebelum perubahan ini) → 404 dengan pesan jelas.

## 4. Format row upload diperluas

Berlaku untuk `POST /datasets` (body `rows`) dan `POST /datasets/upload` (file
.json/.jsonl). Dua field opsional baru per row:

| Field | Wajib? | Perilaku |
|---|---|---|
| `id` | tidak | ID milik pengirim (string/angka), untuk penelusuran balik. Kalau kosong digenerate dari posisi row (0, 1, 2, …). Tidak dipakai proses build. |
| `cve_id` | tidak | ID CVE, mis. `CVE-2019-12111`. Murni provenance. |

## 5. flaw_lines dan func_after boleh bersamaan

Sebelumnya row dengan keduanya ditolak (422). Sekarang boleh:

- `flaw_lines` (anotasi manual) **menang** sebagai ground-truth baris rentan;
- `func_after` tetap disimpan dan menjadi **fallback** (di-diff) untuk row tanpa
  `flaw_lines`.

Aturan lain tetap: row rentan wajib punya minimal salah satu; `flaw_lines` 1-indexed
dan dalam rentang jumlah baris `code`; list kosong bukan anotasi valid.

## Contoh row upload

```json
[
  {"code": "int a;", "label": 0},
  {"id": 257069, "cve_id": "CVE-2017-9350", "code": "...", "cwe": "CWE-125",
   "flaw_lines": [3], "func_after": "..."},
  {"code": "...", "cwe": "CWE-787", "func_after": "..."}
]
```

## Contoh unduh raw

```bash
curl -OJ http://localhost:8000/datasets/{dataset_id}/raw
# -> {raw_id}.json, array JSON, langsung bisa di-upload ulang
```

## File yang berubah

- `API/models/tables.py` — tabel baru `RawDatasetRecord` + FK `datasets.raw_id` +
  kolom `datasets.source_dataset_ids` (schema DB).
- `API/schemas/dataset.py` — field `id` + `cve_id` di DatasetRow, deskripsi flaw_lines.
- `API/services/datasets.py` — tulis `raw.json` kanonik saat job dibuat; parquet build
  ikut membawa kolom `id`/`CVE ID`; larangan both-fields dihapus.
- `API/services/registry.py` — `register_raw`/`get_raw` baru; register_dataset menulis
  `raw_id` + `source_dataset_ids`.
- `API/tasks.py` — ingest hash konten → unggah `{raw_id}.json` → register raw + tautkan
  ke dataset; merge daftarkan `source_dataset_ids`.
- `API/routers/datasets.py` — endpoint `GET /datasets/{dataset_id}/raw` (resolve via
  relasi raw_id).
- `API/openapi.json` — regenerasi (endpoint + field baru).
- `API/README.md` — tabel endpoint, alur ingest, ERD diperbarui.

Kompatibilitas: format upload lama tetap valid (semua field baru opsional). Dataset
lama tetap berfungsi; hanya unduh raw yang 404 untuk mereka.

---

# Perubahan API — Role Datasets (val/test terpisah)

## 6. Dataset terpisah untuk validation dan test (baru)

`POST /train` dan `POST /relearn` menerima dua field opsional:

| Field | Perilaku |
|---|---|
| `val_dataset_id` | Dataset terdaftar dipakai sebagai VALIDATION set. Training memakai 100% `dataset_ids` (tanpa split internal), early stopping menilai di dataset ini. |
| `test_dataset_id` | Dataset terdaftar dipakai sebagai TEST set — mis. **golden benchmark tetap** yang dipakai ulang semua versi model agar metriknya sebanding. Wajib bersama `val_dataset_id`. |

Tanpa keduanya → perilaku lama (split internal val_fraction, test kosong = default prod).

Validasi otomatis (422 bila gagal): label space (num_classes) dan mode harus sama,
featurization (pretrained_lm, func_max_length) harus cocok dengan dataset train,
`config.split` tidak boleh dicampur dengan role datasets.

CIL didukung penuh: pada relearn, validasi label-space memakai aturan **subset** —
benchmark 26 kelas sah terhadap model CIL 36 kelas (target_vocab me-remap-nya).
Kelas role dataset yang tidak ada di class space model → 422. Library memuat tiap
role dataset dengan parameter build miliknya sendiri (`source_val_params` /
`source_test_params`, pola yang sama dengan override per-source di replay), jadi
.pt-nya ketemu meski parameter build task baru berbeda.

Lineage: `val_dataset_id` + `test_dataset_id` disimpan sebagai kolom di `relearn_jobs`
DAN di `models` — dari model langsung terlihat dataset apa yang dipakai untuk val/test.

## 7. Perubahan library (gnn-vuln 0.1.12)

Jalur `source_val`/`source_test` yang sudah ada di library dilonggarkan: sebelumnya
wajib keduanya, sekarang `source_val` saja sah (test kosong, evaluasi test dilewati —
default produksi). Ditambah `source_val_params`/`source_test_params` — override
identitas build per role dataset (max_nodes, filter, sampling, suffix) agar .pt role
dataset ditemukan meski parameter dataset utama berbeda (kasus CIL); tanpa params =
warisi data block config (perilaku lama). Config eksperimen lama tidak berubah
perilaku. **Perlu publish gnn-vuln 0.1.12 + rebuild image worker** sebelum fitur ini
jalan di Docker.

## Contoh

```bash
# golden benchmark: ingest sekali, id-nya dipakai terus
curl -X POST .../datasets/upload -F file=@benchmark.json -F name=golden-benchmark

# train dengan role datasets
curl -X POST .../train -d '{
  "config_id": "graph_based@...",
  "dataset_ids": ["prod_data_abc123"],
  "val_dataset_id": "prod_val_def456",
  "test_dataset_id": "golden_benchmark_789"
}'
```

File berubah (tambahan): `API/schemas/relearn.py`, `API/schemas/train.py`,
`API/services/relearn.py` (build_config + submit), `API/models/tables.py`
(kolom val/test di relearn_jobs + models), `API/services/registry.py`
(register_model), `API/routers/relearn.py`, `API/routers/train.py`,
`src/gnn_vuln/train.py` + `config.py` (lib, val-only + source_val_params/source_test_params), `API/openapi.json`,
`API/README.md` (ERD + alur).

---

# Perubahan API — Multi-Task Continual Sesuai Paper

## 8. ER replay pool kumulatif (perbaikan kesesuaian paper)

Hasil baca ulang paper referensi:

- **EWC-DR (Liu & Chang 2026), Eq (1)**: anchor + importance dari task t−1 SAJA,
  dihitung ulang tiap task. Implementasi kita SUDAH persis ini — tidak diubah.
  (Perlindungan task lama transitif lewat anchor: θ^(t−1) sudah membawa seluruh
  sejarah task sebelumnya.)
- **ER (Chaudhry 2019), Algorithm 1 + ring buffer**: memori episodik KUMULATIF
  lintas SEMUA task lampau, seimbang per kelas. Implementasi lama kita menyimpang —
  buffer hanya dari dataset base terakhir, sampel task-A hilang di update ke-3.

Perbaikan:

- Kolom baru `models.dataset_ids` (JSON) = **lineage data training kumulatif**:
  daftar dataset leluhur + dataset job ini (kronologis, dedup). Diisi otomatis
  saat model diregistrasi — model baru mewarisi daftar milik base model.
- ER/EWC-ER: replay pool = `dataset_ids` kumulatif milik base model. Lebih dari
  satu dataset → digabung dulu (mesin merge .pt yang sudah ada) → buffer 50/kelas
  seimbang di atas gabungan → jaminan ring-buffer per kelas ala Chaudhry terpenuhi
  untuk rantai update sepanjang apa pun. Tanpa perubahan library.
- Kompatibilitas: model lama tanpa `dataset_ids` → fallback `[dataset_id]`
  (perilaku lama, satu generasi).

File berubah (tambahan): `API/models/tables.py` (kolom dataset_ids),
`API/services/registry.py` (register_model), `API/services/relearn.py`
(_cumulative_dataset_ids + replay pool di build_config), `API/README.md` (ERD).
