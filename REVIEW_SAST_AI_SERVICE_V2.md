# Review sast-ai-service — putaran 2

**Tanggal:** 15 Juli 2026
**Repo ditinjau:** `Benardo07/sast-ai-service` @ `ebeac00` (*feat: fix ER cumulative bug*)
**Pin sekarang:** `gnn-vuln==0.1.15`
**Tersedia di PyPI:** `0.1.16` (stabil terbaru), `0.1.17rc1..rc4`

---

## Jawaban singkat

**Naik ke `0.1.16` sekarang.** Satu baris di `pyproject.toml`, tanpa perubahan kode, dan langsung membuka satu kemampuan yang hari ini kalian tolak sendiri lewat `ValueError`.

**Jangan naik ke `0.1.17` dulu.** `0.1.17` akan **menggagalkan setiap relearn** di image kalian yang sekarang — bukan melambat, tapi mati dengan `FileNotFoundError`. Sebabnya di bagian 2. Ada 3 hal yang harus dikerjakan dulu; semuanya kecil, tapi tak boleh dilewat.

---

## 1. Yang sudah beres — konfirmasi

Tiga temuan merah dari review pertama sudah benar-benar diperbaiki, bukan ditambal:

| Temuan lama | Status |
|---|---|
| Fisher EWC dihitung di task-B | **Beres** — `importance_source` menunjuk dataset t-1 saja (`relearn.py:262-267`), sesuai EWC-DR Eq(1) |
| `raw_func` kosong untuk hybrid + sequential | **Beres** — `_write_inline_dataset` selalu menulis `raw_func` (`relearn.py:807-810`) |
| `class_names` saat serving | **Beres** — `target_vocab` diturunkan dari `base_class_names`, dan replay pool gabungan **menolak** jalan tanpa itu (`relearn.py:243-248`). Penolakan itu keputusan yang tepat: tanpa `target_vocab`, buffer replay memberi graph yang benar dengan label yang salah — rusak diam-diam. |

---

## 2. BLOKIR untuk 0.1.17 — filter XML jadi error keras

Ini yang paling penting di dokumen ini.

**Fakta 1.** Config ketiga model produksi (yang kami kirim bersama checkpoint) memuat:
```yaml
data:
  filter_top25_dangerous: true
```

**Fakta 2.** File XML-nya (`data/cwe/top25.xml`) **tidak pernah sampai ke container kalian**. `Dockerfile` cuma menyalin `pyproject.toml` + `app/`, dan wheel `gnn-vuln` cuma memaketkan `src/gnn_vuln` (`only-include`). Jadi `data/cwe/` tidak ada di image mana pun.

**Fakta 3.** Sampai `0.1.16`, XML yang hilang cuma menghasilkan *warning* lalu mengembalikan himpunan kosong — **filternya mati diam-diam**. Kebetulan itulah yang membuat layanan kalian jalan hari ini.

**Yang berubah di `0.1.17`:** filter diminta tapi XML tak ada = **`FileNotFoundError`**, sengaja dibuat keras. Alasannya: "diam-diam tidak memfilter" itu justru cara paling mahal untuk salah — dataset yang kalian kira sudah disaring ternyata tidak, dan namanya pun ikut berbeda.

**Akibatnya kalau kalian naik sekarang:** setiap `/relearn` dan `/train` mati saat memuat dataset.

**Perbaikannya, pilih salah satu:**

- **Disarankan** — buang `filter_top25_dangerous`, `filter_owasp`, `top_cwe`, `max_per_class` dari blok `data` config yang kalian kirim ke library. Untuk layanan, knob ini memang tak masuk akal: baris data milik pengguna, jadi **semua kelas** yang ada di dalamnya harus dibangun. Menyaring "top 25" itu kebiasaan benchmark, bukan kebiasaan produksi. Di `0.1.17` kontrak serving (`gnn_vuln.serving`) memang sudah membuang keempatnya, dan penyempitan hanya boleh lewat `cwe_list` yang eksplisit dan bisa diaudit.
- Atau — salin `data/cwe/*.xml` ke dalam image. Ini menyembuhkan errornya, tapi **mengubah nama dataset turunan** (lihat bagian 3), jadi cache `.pt` lama kalian tak akan ketemu lagi.

---

## 3. Bom waktu yang sudah aktif — komentar kalian salah

Di `relearn.py:653` tertulis:

> `filter_owasp/filter_top25_dangerous` **ARE forwarded and merge fine.**

**Tidak.** Yang benar: keduanya *diteruskan*, tapi nama `.pt` hasil merge tetap melenceng. Sampai `0.1.16`, `_out_processed_path` menghitung hash suffix dari `cwe_list` **mentah** (kosong), sedangkan `dataset_lm` — yang dipakai trainer — memperluas XML **lebih dulu** lalu menghitung hash dari daftar CWE hasil perluasan. Dua sisi, dua nama:

```
merge menulis  : ..._f22b93ac1...
trainer mencari: ..._f40f2e964...   ← nama dataset resmi kami
```

Merge berfilter **selalu** menulis nama yang tak pernah dicari siapa pun. Kami baru menemukannya sendiri bulan ini (dan itu bug lama, bukan bug baru).

Kenapa kalian belum kena? **Kebetulan** — karena XML tak ada di image kalian, kedua sisi sama-sama merosot ke daftar kosong, jadi namanya kebetulan cocok. Artinya: **hari pertama seseorang me-mount `data/cwe/`, replay pool gabungan kalian langsung rusak** — dan rusaknya senyap, trainer cuma akan mencoba membangun ulang dari CPG mentah.

Diperbaiki di `0.1.17` (perluasan XML jadi satu fungsi bersama, dipakai merge dan `dataset_lm`). Ini alasan kuat untuk akhirnya naik ke `0.1.17` — tapi baru **setelah** bagian 2 dibereskan.

---

## 4. Yang langsung kalian dapat dari 0.1.16

`_write_merge_config` (`relearn.py:645-663`) menolak mentah-mentah config yang punya `cwe_list` atau `cwe_groups`:

```python
raise ValueError("cannot merge a cumulative replay pool for a config with data.cwe_list or ...")
```

Penjelasan kalian benar untuk `0.1.15`: `_filter_suffix(None, None, ...)` memang menjatuhkan keduanya. **Sudah diperbaiki di `0.1.16`** — `_filter_suffix` sekarang meneruskan `cwe_list` + `cwe_groups`, dan `_build_ds` juga meneruskan `top_cwe` / `max_per_class` / `resample_seed`.

Jadi begitu pin naik ke `0.1.16`, **hapus `ValueError` itu**. Replay pool kumulatif untuk config berfilter jalan.

---

## 5. Bug yang berdiri sendiri — tak tergantung versi

### 5.1 `_parse_metrics` bisa memasang metrik **model lain** (paling serius)

`relearn.py:345-365`:

```python
search_dirs = [checkpoint.parent, results_dir, settings.results_root]
...
metrics.update(self._scalar_metrics(data))   # ← dir TERAKHIR yang menang
```

`settings.results_root` itu direktori **global**, dan yang diambil adalah `metrics_summary.json` **termuda** di bawahnya — milik run mana pun. Karena `update()` bersifat last-wins, isinya menimpa metrik milik job itu sendiri.

Kapan meledak: **saat run tidak menghasilkan metrik sama sekali**. Kalau `1 - train_ratio - val_ratio == 0` (tidak ada test split), library melewati evaluasi dan tidak menulis `metrics_summary.json`. Job ini tak punya metrik → fallback ke `results_root` → mengambil metrik run **sebelumnya** → dan model baru terdaftar dengan **angka milik model lain**. Tanpa error, tanpa warning.

Ini bukan hipotetis: `RelearnSplit.train_ratio/val_ratio` boleh diisi bebas oleh pemanggil, jadi `0.9/0.1` (default produksi yang wajar) sudah cukup memicunya.

**Perbaikan:** baca **hanya** direktori milik job itu. Kalau tak ada metrik, biarkan `None` dan katakan alasannya di `job.message`. Di `0.1.17` library sudah memberi pesan eksplisit persis untuk kasus ini:

> `registered WITHOUT metrics: no test set. Pass test_dataset_id (a fixed benchmark, comparable across versions), or leave 1 - train_ratio - val_ratio > 0 to hold one out.`

Model tanpa metrik itu sah. Model dengan metrik **milik orang lain** itu racun — dan itu yang dipakai orang untuk memutuskan promosi ke produksi.

### 5.2 Semuanya ditemukan lewat `glob` + `mtime`

Tiga tempat: checkpoint (`:288`), metrik (`:356`), dataset (`:537`). Pola `sorted(..., key=mtime)[0]` itu tebakan, bukan referensi. Ia benar selama tak ada yang berjalan bersamaan dan tak ada artefak nyasar — dua syarat yang tak bisa kalian jamin selamanya.

`0.1.17` memberi jalan keluar: setiap run menulis **`run_result.json`** (bisa diarahkan lewat env `GNN_VULN_RUN_RESULT`) berisi path checkpoint, path dataset, metrik, dan class_names — hasilnya **diberitahukan**, bukan ditebak.

### 5.3 `cwe_vocab.json` sebagai sumber ruang kelas

Dipakai di `loader.py:124`, `relearn.py:179/331/465`. Masalahnya: bundle hasil merge **menyusun ulang** vocab (benign dulu, sisanya urut), jadi `cwe_vocab.json` di sebelah dataset tidak selalu = urutan kelas yang dipakai model.

Kebenarannya ada di `.pt` (`class_names` di meta), bukan di file JSON di sebelahnya. Di `0.1.17`, `core.open_dataset(path)` mengembalikan `DatasetInfo.class_names` langsung dari `.pt`, dan `cwe_vocab.json` sudah **hilang total** dari jalur API kami — vocab jadi nilai di memori saat build, bukan file.

### 5.4 "TEST butuh VAL"

Komentar di `schemas.py:111` benar untuk `0.1.15/0.1.16`: test-only diabaikan library. **Sudah tidak berlaku di `0.1.17`** — test jadi peran mandiri, boleh dicampur (rasio untuk train/val + dataset benchmark tetap untuk test). Itu justru bentuk yang kalian inginkan: satu benchmark tetap, dibandingkan lintas versi model.

---

## 6. Rencana naik versi

**Langkah 1 — sekarang, aman, 1 baris:**
```toml
"gnn-vuln==0.1.16",
```
lalu hapus `ValueError` di `_write_merge_config` (bagian 4). Perbaiki juga 5.1 — itu independen dari versi dan yang paling berbahaya di antara semuanya.

**Langkah 2 — `0.1.17`, setelah 3 hal ini:**

1. **Buang knob benchmark** dari blok `data` yang kalian kirim (`filter_top25_dangerous`, `filter_owasp`, `top_cwe`, `max_per_class`) — kalau tidak, setiap run mati (bagian 2).
2. **Ganti `GNN_VULN_API_MODE`.** Env itu **dihapus** di `0.1.17`. Kode kalian menyetelnya di `relearn.py:391` + `:781` dan sengaja melepasnya di `:971`. Setelah upgrade env itu jadi **tak berefek** — dan gagalnya senyap: tiap run kembali menulis artefak riset (`predictions.csv`, `embeddings.npz`, plot, `training_log.csv`) ke disk. Gantinya dua field yang ikut tersimpan bersama config run:
   ```yaml
   train:
     research_artifacts: false   # skip CSV/kurva/plot
   data:
     no_build: true              # .pt hilang = error, BUKAN diam-diam membangun ulang lewat Joern
   ```
   `no_build` itu jaring pengaman yang selama ini tak kalian punya: layanan hanya mengirim `.pt` yang sudah jadi, jadi `.pt` yang tak ketemu adalah **bug**, bukan aba-aba menyalakan Joern selama berjam-jam.
3. **Berhenti mengimpor `_out_processed_path`** (`relearn.py:758`). Itu fungsi privat, dan semantik namanya **berubah** di `0.1.17`. Penggantinya publik dan tak menebak nama sama sekali:
   ```bash
   python -m gnn_vuln.core merge-datasets --paths a.pt b.pt --out merged.pt --result-json info.json
   ```
   `info.json` berisi path, `class_names`, `n_graphs`. Lalu suapkan ke trainer lewat `data.dataset_path` — **path**, bukan nama yang diturunkan dari parameter. Ini menutup seluruh kelas bug "nama tak cocok" selamanya, termasuk bagian 3.

---

## 7. Catatan arah, bukan bug

Yang di bawah ini **kami** juga masih punya di API sendiri, dan sedang kami benahi — sampaikan biar kontraknya tidak makin jauh berbeda:

- **Merge di dalam `/relearn` sebaiknya mati.** Sekarang `dataset_ids` (list) digabung otomatis di dalam, dan `.pt` gabungan itu **tidak pernah terdaftar** sebagai entitas: tanpa id, tanpa config beku, tanpa fingerprint. Artinya data yang **benar-benar melatih model** tidak punya identitas — cuma daftar bahan. Bentuk yang benar: merge jadi langkah eksplisit yang menghasilkan dataset terdaftar (punya id + provenance), lalu relearn menerima **satu id per peran**.
- **Fingerprint.** Di `0.1.17` tiap `.pt` menyimpan identitas build di dalam metanya (LM, dimensi, versi lib). Model bisa **menolak** dataset yang tak cocok dalam 0,1 detik **sebelum** bundle diunduh. Nama file itu label, bukan jaminan — dan selama ini kita berdua bergantung pada label.

---

*Ditulis dari sisi library. Semua rujukan baris mengacu ke `ebeac00`.*
