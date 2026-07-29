# Revisi pasca-simulasi sidang

Feedback dari simulasi thesis defense bersama dosen pembimbing. Lima isu, plus dua hal
yang perlu dikonfirmasi dulu sebelum eksekusi. Belum ada perubahan dokumen.

## Perlu dikonfirmasi dulu

- [x] **Provenance graph MegaVul utama. DIKONFIRMASI.** Dataset utama memakai struktur CPG
      bawaan MegaVul (kolom `func_graph_path`), karena MegaVul membangunnya dengan Joern
      yang sama sehingga bisa langsung dipakai. Fitur node (embedding UniXcoder, jarak,
      penanda API) dihitung sendiri, cpg14 = pangkas edge dari CPG bawaan itu. BigVul dan
      dataset lain (termasuk continual domain baru) dibangun sendiri dengan Joern.
      Joern tetap ada di pipeline untuk dua keperluan, yaitu dataset non-MegaVul dan
      integrasi SAST tempat graph tidak tersedia di awal sehingga harus digenerate live.
      (Poin live-Joern ini menguatkan tautan #4 ke kebutuhan SAST.)
- [x] **Rujukan capstone. DIKONFIRMASI.** Pakai generik "(laporan capstone)" karena
      capstone belum selesai. Isi capstone = kebutuhan, rancangan, hingga pengujian
      (setara dokumen teknis), jadi boleh dirujuk untuk kebutuhan sistem SAST.

## Urutan kerja (setelah approval)

1. #2 provenance graph — SELESAI
2. #3 + #5 BERSAMA — SELESAI (anchor realisme IV.2.1.1, metrik IV.2.2.1/2, 3 referensi)
3. #1 motivasi eksperimen — SELESAI (peta IV.4 opener + klausa biaya SAST cpg14)
4. #4 tautan capstone — SELESAI, digabung #1 (anchor capstone di III.1 opener + III.1.2)

SEMUA 5 ISU SELESAI. Ringkasan tautan SAST tersebar, yaitu III.1 opener (anchor capstone
3 kebutuhan), III.1.2 (sistem SAST dikembangkan), IV.2.1.1 (realisme utk capstone),
IV.2.2.1 (macro utk semua CWE), IV.2.2.2 (effort reviewer), IV.4 opener (peta eksperimen
ke RM+SAST), IV.4.2 (biaya cpg14). Tak repetitif, tiap titik gagasan beda.

Benang merah #2/#3/#5, yaitu MegaVul real-sourced (CVE/NVD + fix commit nyata, bukan
sintetis) -> distribusi (rasio rentan + long-tail jenis CWE) cerminkan realitas terlapor.
Argumen ditulis SEKALI di IV.2.1.1, dirujuk #3, biar tak repetitif.

---

## #1 — Motivasi eksperimen ke RM + kebutuhan SAST

**Status docs:** Eksperimen utama (perbandingan antararsitektur, lawan baseline) sudah
dibingkai RM1/RM2 di pembuka IV.4.1/IV.4.2. Eksperimen analisis tidak punya kalimat
motivasi, yaitu cpg14, call-collapse, API-bias, mode-panjang muncul sebagai temuan tanpa
"kenapa diuji".

**Usulan:**
- [ ] Paragraf peta di pembuka IV.4 (`diskusi_hasil.md`) memetakan tiap eksperimen -> RM
      mana -> kebutuhan SAST mana.
- [ ] Klausa motivasi 1 kalimat di tiap eksperimen analisis:
  - cpg14 -> biaya bangun CPG penuh terbayar atau tidak, penting bagi SAST yang memindai
    banyak fungsi
  - call/API -> sudah jadi bukti taint RM1, eksplisitkan tautannya
  - mode-panjang -> keandalan SAST lintas ukuran fungsi
  - ambang -> sudah ada (kendali operasional SAST), biarkan

## #2 — Graph bikin sendiri vs pakai jadi (MegaVul)

**Status docs:** Inkonsisten. bab-4 baris 11 = pakai CPG jadi bawaan MegaVul (hindari
jalankan Joern). bab-3 III.2.1 baris 3 = "diubah menjadi CPG menggunakan Joern" (seakan
bikin sendiri). Continual (bab-4 baris 115) = bikin sendiri.

**Usulan:** SELESAI.
- [x] bab-3 III.2.1, yaitu tambah kalimat CPG bagian dari pipeline sub-sistem sehingga
      graph dibangun langsung dari kode masuk saat integrasi SAST (view desain/live).
- [x] bab-4 IV.2.1.1, yaitu perjelas MegaVul pakai graph bawaan (Joern sama, tak bangun
      ulang), struktur dari bawaan + fitur node dihitung sendiri (rujuk III.2.1), dataset
      lain di luar MegaVul bangun CPG sendiri dengan Joern yang sama.
- Angle live-Joern SAST untuk #4 sudah tersemai di bab-3, tinggal diperkuat saat #4.

## #3 — Justifikasi metrik utama (macro F1, lokalisasi) [KAIT #5]

**Status docs:** macro F1 sudah dijustifikasi (`prosedur_dan_data.md` baris 192, 211) tapi
framing statistik saja (imbalance -> weight sama, Grandini 2020). Belum ada kenapa bukan
precision/recall tunggal, belum tertaut kebutuhan SAST, dan belum jawab pertanyaan penguji
soal tail menggeser macro drastis.

**Usulan (IV.2.2.1, RUJUK anchor realisme #5, jangan ulang argumen realisme):**
- [ ] macro = kemampuan per jenis (CWE langka-tapi-parah terlewat = pelanggaran nyata,
      SAST wajib mampu tiap jenis). weighted-F1 + akurasi = performa pada prevalensi nyata
      (beban alarm). Dua pertanyaan beda, itu sebab keduanya dilaporkan.
- [ ] Kenapa bukan precision/recall tunggal (precision-saja hadiahi model diam, recall-saja
      hadiahi tandai-semua; F1 harmonik seimbangkan alarm palsu vs kerentanan terlewat,
      dua-duanya penting reviewer SAST).
- [ ] Jawab pertanyaan penguji (tail geser macro), yaitu akui trade-off sadar, tail-
      sensitivity = harga agar kebutaan satu kelas tak tersembunyi. Diredam multi-seed
      mean±std + aturan <2×std + pendamping + ancaman ketujuh (tail data-capped). Rujuk
      anchor #5 untuk "tail memang langka di realitas, bukan artefak".
- [ ] IV.2.2.2 lokalisasi, yaitu tautkan IFA/Top-5 ke effort reviewer SAST eksplisit.

## #4 — Laporan terlalu individu, tak tertaut capstone/kebutuhan SAST

**Status docs:** bab-1 I.3 sudah sebut "sub-sistem bagian dari SAST capstone". Keputusan di
bab-3/bab-4 hanya di-back paper, bukan kebutuhan sistem SAST.

**Usulan (refer capstone generik, TIDAK ke lampiran/dokumen teknis):**
- [ ] Sisipkan tautan kebutuhan SAST di titik keputusan kunci:
  - Tiga keluaran (cls+loc+continual) -> kebutuhan SAST akan keluaran dapat ditindaklanjuti
    + dapat diperbarui
  - Granularitas fungsi -> titik integrasi SAST
  - Weakly-supervised loc -> SAST harus jalan di semua fungsi tanpa label baris
  - macro F1 -> SAST wajib tangani semua jenis CWE
- [ ] Bingkai pembanding baseline (IV.4 kesimpulan) sebagai "tiap baseline kehilangan
      sesuatu yang dibutuhkan sistem SAST", bukan cuma trade-off akademik.

## #5 — Kedekatan dataset dengan realitas [ANCHOR, fondasi #2 + #3]

**Status docs:** Tabel IV.1 rasio nyata (375rb tak-rentan vs 20rb rentan ~1,8% rentan),
lalu cap tak-rentan ke 1.600. Ancaman ketujuh sebut "satu sumber MegaVul". Belum ada
argumen eksplisit real-sourced -> distribusi cerminkan realitas, dan belum bahas apakah
tail CWE dataset juga tail di realitas.

**Usulan — tulis SEKALI di IV.2.1.1 (dekat Tabel IV.1 + IV.5), jadi anchor #2/#3:**
- [ ] Alasan pilih MegaVul, yaitu real-sourced (CVE/NVD + fix commit nyata, bukan sintetis
      seperti SARD/Juliet) -> cerminkan realitas -> cocok latih SAST deploy nyata. (Ini
      sekaligus alasan pemilihan #2.)
- [ ] Kedekatan jenis CWE, yaitu long-tail CWE MegaVul ~ prevalensi terlapor. C/C++
      didominasi bug memori (= head kita CWE-787/125/476/416), CWE langka (CWE-639) memang
      jarang di CVE nyata. Antar-kelas CWE TAK diseimbangkan -> tail dataset = tail
      realitas, bukan artefak koleksi.
- [ ] Caveat jujur:
  - rasio rentan:tak-rentan diseimbangkan untuk latih (!= ~1,8% realitas) -> kalibrasi FP
    tergantung deployment, ambang confidence (IV.4.4) jadi penyetel
  - NVD = subset terlapor/terdisclose, bias proyek pen-CVE -> proxy wajar bukan sama persis
  - satu korpus (ancaman ketujuh, rujuk)
  - peringkat model + kemampuan (cls/loc/continual) tetap valid, hanya kalibrasi FP
    bergantung distribusi deployment

**Referensi tambahan (KETEMU, tambahkan ke referensi.md saat eksekusi):**
- Dominasi memory-safety pada C/C++ nyata (back klaim head = memori = realitas):
  - Microsoft/Miller (MSRC 2019), yaitu ~70% CVE Microsoft selama 12 tahun = memory
    safety. https://msrc.microsoft.com/blog/2019/07/a-proactive-approach-to-more-secure-code/
  - Chromium memory-safety, yaitu ~70% bug high-severity = memory unsafety atas 912 bug
    sejak 2015, ~35% use-after-free (= CWE-416), sisanya OOB (= CWE-125/787).
    https://www.chromium.org/Home/chromium-security/memory-safety/
- Long-tail distribusi JENIS kerentanan (back klaim tail langka wajar, bukan artefak):
  - Liu, Meng, Zou, Gong, Li, Lin, Sun, Huo, Zhang (2020), A Large-Scale Empirical Study
    on Vulnerability Distribution within Projects and the Lessons Learned, ICSE '20.
    Temuan, yaitu ~70% kerentanan jatuh pada ~20% jenis, top-3 jenis ~50%.
    DOI 10.1145/3377811.3380923

Pemakaian, yaitu Microsoft + Chromium untuk "C/C++ nyata didominasi bug memori = head
kita (CWE-787/125/476/416)", Liu dkk untuk "distribusi jenis long-tail = tail dataset
cerminkan realitas". Chromium C/C++-spesifik, jadi paling pas; Liu dkk = akademik peer-
reviewed untuk klaim long-tail umum.

---

# Revisi PASCA-SIDANG (belum dikerjakan)

Ditemukan sendiri saat menyiapkan slide, bukan dari feedback pembimbing. Laporan sudah
dikumpulkan, jadi JANGAN diubah sebelum sidang. Kerjakan setelahnya.

## P#1 — Gambar III.2 alur pipeline, notasi activity diagram perlu dirapikan

**Berkas:** `docs/laporan-individu/image/bab-3/alur_pipeline.png`
**Dipakai di:** `bab-3/konsep_solusi.md` baris 12 (Gambar III.2), `slides/03_solusi.md` baris 8

### Inventaris bentuk saat ini (sudah diperiksa dengan zoom)

Kabar baiknya, aturan bentuk SUDAH konsisten. Membulat dipakai untuk aksi, siku untuk
object node.

| Kotak | Bentuk | Status |
| --- | --- | --- |
| Bangun CPG dengan Joern | membulat | aksi, benar |
| Sematkan embedding UniXcoder dan fitur struktural | membulat | aksi, benar |
| Tokenisasi teks fungsi | membulat | aksi, benar |
| Model AI | membulat | SALAH, nomina di kotak aksi |
| Source code fungsi | siku | object node, benar |
| Enriched CPG | siku | object node, benar |
| Token teks fungsi | siku | object node, benar |
| Jenis kerentanan (CWE) | siku | object node, benar |
| Baris penyebab kerentanan | siku | object node, benar |

### Aturan UML 2.5 yang dipakai sebagai acuan perbaikan

1. **Action** = kotak sudut membulat, labelnya frasa kerja.
2. **Object node** = kotak sudut siku, labelnya nomina. Boleh ditulis `nama : Tipe`, dan
   keadaannya dalam kurung siku, misalnya `Enriched CPG [siap latih]`.
3. **Initial node** = lingkaran hitam penuh. **Activity final** = lingkaran bermata.
   **Flow final** = lingkaran bersilang, menghentikan satu cabang saja.
4. **Fork dan join** = batang tebal. Fork satu masuk banyak keluar, join banyak masuk satu
   keluar. Jangan memakai batang untuk percabangan bersyarat, itu tugas decision node.
5. **Decision dan merge** = belah ketupat, guard ditulis dalam kurung siku di edge.
6. **Control flow tidak boleh berujung pada object node**, kecuali object node itu bertipe
   kontrol. Aliran yang menyentuh object node adalah **object flow**. Di UML 2.x keduanya
   sama-sama panah garis penuh, jadi bedanya bukan pada gambar edge melainkan pada apa yang
   disambungkannya.
7. **Masukan dan keluaran aktivitas** digambar sebagai **activity parameter node**, yaitu
   object node yang menempel pada bingkai aktivitas. Bukan initial node yang menunjuk ke
   sebuah object node.
8. **Object node dengan beberapa edge keluar berarti token pergi ke SALAH SATU**, bukan ke
   semuanya. Untuk menggandakan token wajib lewat fork. Diagram ini sudah benar di titik itu.
9. **Pengelompokan langkah**, pilihannya structured activity node (kotak putus-putus dengan
   keyword «structured») atau activity partition berupa swimlane bergaris penuh dengan nama
   pada headernya. Kotak putus-putus bersudut membulat tanpa keyword bermakna
   interruptible activity region, yang bukan maksud di sini.

### Daftar perbaikan

**A. Wajib. "Model AI" nomina di kotak aksi. SUDAH DIKERJAKAN untuk slide.**
Labelnya jadi "Prediksi dengan model AI". Sudah diterapkan pada berkas slide
`image/bab-3/alur_pipeline_slide.png` (sumber `alur_pipeline_slide.drawio`), yang sekaligus
digambar ulang mendatar agar muat di slide. Berkas laporan `alur_pipeline.png` SENGAJA
dibiarkan menegak dan berlabel lama karena laporan sudah dikumpul. Saat revisi pasca-sidang,
tinggal ganti berkas laporan mengikuti versi slide.

**B. Sebaiknya. Masukan dan keluaran pakai activity parameter node.**
Sekarang initial node menunjuk langsung ke object node "Source code fungsi", dan dua object
node keluaran menunjuk ke join lalu activity final. Keduanya membuat control flow menyentuh
object node, melanggar aturan 6. Perbaikannya, gambar "Source code fungsi", "Jenis
kerentanan (CWE)", dan "Baris penyebab kerentanan" sebagai object node yang menempel pada
bingkai aktivitas, lalu initial node langsung ke fork.

**C. Sebaiknya. Kotak "Preprocessing".**
Sekarang kotak putus-putus tanpa keyword. Tambahkan «structured», atau ubah jadi partition
bergaris penuh, supaya tidak terbaca sebagai interruptible activity region.

**D. Opsional. Fork setelah model.**
Dua keluaran dari satu aksi lebih tepat digambar sebagai dua output pin pada aksi itu
daripada fork bar. Fork bar tetap terbaca, jadi ini bukan kesalahan, hanya kurang presisi.

### Yang JANGAN diubah, sudah benar

- Fork setelah "Source code fungsi" memang wajib ada, karena object node dengan dua edge
  keluar tanpa fork berarti token cuma pergi ke salah satu cabang.
- Join sebelum model benar, kedua cabang memang harus selesai dulu.
- Bentuk membulat untuk aksi dan siku untuk object node sudah konsisten, pertahankan.

### Saat sidang

Kemungkinan besar tidak ditanya, karena cacatnya tinggal satu label. Kalau ditanya, akui
"Model AI" semestinya frasa kerja seperti "Prediksi dengan model AI", dan sebut sudah masuk
daftar revisi. Jangan mengarang pembenaran.

### Alat menggambar ulang

Gambar sekarang keluaran draw.io. PlantUML activity beta tidak punya object node yang
sungguhan, jadi tetap pakai draw.io saja untuk diagram ini.

## P#2 — Bab-5 mengutip angka satu arsitektur tanpa menyebutnya (RINGAN, opsional)

**Berkas:** `docs/laporan-individu/bab-5.md`, paragraf rumusan kedua

**Sudah diperiksa dan TERNYATA BUKAN MASALAH, jangan diubah.**
Rumusan ketiga aman. Pemilihan arsitektur untuk eksperimen continual sudah dinyatakan
eksplisit beserta alasannya di Subbab IV.3.4, yaitu "memakai arsitektur berbasis graph yang
paling ringan secara komputasi". Bab V meringkas hasil itu, tidak perlu mengulang
pemilihannya. Kalau ditanya saat sidang, tunjuk kalimat itu.

**Yang tersisa, ringan.** Paragraf rumusan kedua menulis "Top-1 sekitar 0,257 dan median
IFA sekitar tiga baris" tanpa menyebut arsitekturnya. Angka itu milik arsitektur berbasis
graph, sedangkan Tabel IV.12 memuat hibrida 0,245 dan sekuensial 0,273. Kata "sekitar"
sudah melunakkannya dan 0,257 berada di tengah rentang, jadi ini bukan pemilihan angka yang
menguntungkan. Tetap saja pembaca tidak bisa tahu angka siapa yang dikutip.

**Perbaikan yang diusulkan, pilih salah satu.**

- **Pilihan A, paling ringkas.** Tambahkan "pada arsitektur berbasis graph". Satu frasa,
  tidak mengubah angka mana pun.
- **Pilihan B, rentang ketiganya.** Top-1 0,245 sampai 0,273, median IFA 3 sampai 4 baris,
  fungsi tanpa anotasi 0,63 sampai 0,65. Lebih mewakili, tetapi angkanya jadi kabur saat
  diucapkan dan Bab V memang bergaya menyebut satu angka wakil.

**Tambahan yang bukan cacat, tetapi menguntungkan.** Bab V tidak pernah merujuk Lampiran I
dan Lampiran J, padahal keduanya memperlihatkan pola continual yang sama berulang pada
arsitektur sekuensial dan hibrida. Satu klausa di paragraf rumusan ketiga akan memakai
lampiran yang sudah terlanjur ditulis, sekaligus memperkuat kesimpulannya.

**Saat sidang.** Kalau ditanya "angka itu arsitektur yang mana", jawab arsitektur berbasis
graph dan sebutkan dua angka lainnya dari Tabel IV.12. Untuk continual, tunjuk IV.3.4 yang
sudah menyatakan pemilihannya, lalu sebut Lampiran I dan J sebagai pembuktian pada dua
arsitektur lain. Slide penutup sudah memuat keduanya.

## P#3 — LAPORAN CAPSTONE, frasa "makna dan konteks kode" mudah disalahbaca

**Berkas:** `docs/laporan-capstone/bab-1.md`, paragraf pertama dan paragraf keempat

**Masalah.** Bab-1 memakai frasa "makna dan konteks kode" dua kali, sekali sebagai hal yang
sulit dikenali aturan tetap, sekali sebagai nama kebutuhan pertama. Frasa itu terbaca seperti
dua kemampuan terpisah, dan pembaca yang juga membaca laporan individu akan menduga pemetaan
"makna sama dengan LM, konteks sama dengan GNN".

Pemetaan itu tidak pernah dibuat laporan mana pun, dan bertentangan dengan contoh yang bab-1
berikan sendiri, yaitu kerentanan yang muncul dari alur kontrol dan dependensi data. Contoh
itu justru wilayah graph, bukan LM.

**Bukan salah tulis, hanya rawan.** Kalimatnya benar dan bersitasi Shaon & Akter (2025).
Yang kurang adalah penegasan bahwa itu satu gagasan.

**Perbaikan yang diusulkan.** Pada paragraf keempat, tempat ketiga kebutuhan disebut, ganti
nama kebutuhan pertama menjadi rumusan yang tidak bisa dipecah dua, misalnya "deteksi
kerentanan yang mempelajari pola langsung dari data alih-alih aturan yang ditulis manual".
Rujukannya sudah ada di paragraf pertama, yaitu Harzevili et al. (2019). Paragraf pertama
biarkan apa adanya, karena di sana frasa itu memang menerangkan keterbatasan aturan tetap
dan langsung diikuti contohnya.

**Sudah diterapkan di slide.** Deck ringkas dan deck panjang capstone memakai rumusan baru
untuk nama kebutuhan, sedangkan kutipan bersitasi pada bullet latar belakang tetap memakai
frasa aslinya. Catatan penyaji pada deck ringkas memuat peringatan agar tidak memetakan
makna ke LM dan konteks ke GNN.

**Catatan penting, jangan salah paham.** GNN, LM, dan Code Property Graph MEMANG bagian
rumusan masalah capstone. RM 1 menyebutnya eksplisit, yaitu "modul deteksi kerentanan berbasis
AI dengan Graph Neural Network dan Language Model atas Code Property Graph". Yang generik
hanyalah paragraf KEBUTUHAN, dan itu wajar, yaitu kebutuhan dulu lalu pendekatan.

**Saat sidang.** Sebut GNN, LM, dan CPG dengan bebas, karena ada di rumusan masalah sendiri.
Kalau digali lebih jauh, sebut mekanismenya lugas, yaitu memproses Code Property Graph yang
memuat alur kontrol dan dependensi data, dan LM dipakai pada DUA tingkat, yaitu memperkaya
tiap node CPG dan membaca fungsi secara utuh lewat jalur LM tingkat fungsi. Jangan menyebut
LM hanya untuk embedding node. Lalu tunjuk laporan individu untuk evaluasinya. Yang tidak dipakai di forum capstone hanya istilah
struktural berbanding kontekstual, dan perbandingan ketiga arsitektur.

## P#4 — Kolom n test, dua cacat dan satu penjelasan yang perlu ditulis

**Berkas:** `docs/laporan-individu/bab-4/hasil_evaluasi.md`, Tabel IV.10, IV.11, IV.12, IV.13

Seluruhnya **SUDAH DIVERIFIKASI** dari artefak per seed, sebagian diunduh dari Drive
`gdrive-mesach:tugas-akhir/results/baselines/` dan `data/baselines/`.

### Cacat 1, angka LIVABLE salah

Tabel IV.10 menulis n LIVABLE **533**. Yang benar **576**, konstan pada ketiga seed.

Terverifikasi dari `livable_megavul_s{42,1,2}_*_adamw1e-3` yang metriknya cocok persis dengan
laporan, yaitu macro F1 0,0421, 0,0490, 0,0328 dengan mean 0,041 ± 0,008 sesuai Tabel IV.10 dan
Lampiran D.3, serta akurasi 0,2587, 0,3247, 0,2986 dengan mean 0,294 ± 0,033. Ketiganya
melaporkan `n = 576`. Angka 533 tampaknya tersisa dari run lama.

### Cacat 2, empat sel kehilangan standard deviation

| Tabel | Model | Per seed 42, 1, 2 | Seharusnya |
| --- | --- | --- | --- |
| IV.10 | VulExplainer | 914, 910, 921 | **915 ± 6** |
| IV.11 | Usulan | 590, 578, 603 | **590 ± 13** |
| IV.12 | Usulan | 581, 578, 593 | **584 ± 8** |
| IV.12 | LineVul | 402, 418, 422 | **414 ± 11** |

Yang lain memang konstan, jadi biarkan tanpa ±, yaitu usulan 25 kelas 913, usulan 26 kelas
1.073, LOSVER 478 dan 476, LineVD 599, LIVABLE 576.

### Penjelasan yang perlu ditulis, kenapa sebagian konstan

**Logika split-nya sama untuk semua model.** Seluruh baseline memakai ekspor dari
`scripts/export_baseline_split.py`, yang mereplikasi `dataset_lm.get_splits`, yaitu shuffle
berseed lalu 80/10/10. Ekspor per seed memang ada di Drive, yaitu
`megavul_ml1024_baselines_20260707` untuk seed 42, `megavul_ml1024_split_s1`, dan
`megavul_ml1024_split_s2`. Ketiganya berukuran sama, yaitu test 1.073, tetapi anggotanya
berbeda, dengan irisan seed 42 dan seed 1 hanya 103 dari 1.073.

Jadi yang menentukan konstan atau tidak bukan seed-nya, melainkan **apa yang dihitung**.

| Jenis | Contoh | Konstan? |
| --- | --- | --- |
| Ukuran split itu sendiri | usulan 26 kelas 1.073 | ya, 10% dari dataset berukuran tetap |
| Ukuran split dataset vuln-only | usulan 25 kelas 913 | ya, seluruh isi dataset rentan |
| Hasil saringan atas isi split | VulExplainer `vul == 1` | tidak, 914, 910, 921 |
| Hasil saringan atas isi split | usulan lokalisasi, baris jatuh ke node | tidak, 581, 578, 593 |

Angka VulExplainer 914, 910, 921 **cocok persis** dengan banyaknya fungsi rentan pada ketiga
ekspor split, jadi baseline itu memang memakai split per seed.

**Yang belum tuntas, SUDAH DICEK tetapi tidak menyimpulkan.** LOSVER 478 dan LineVD 599
konstan pada ketiga seed, padahal LineVul dan VulExplainer berubah.

Banyaknya fungsi rentan ber-`flaw_lines` pada ketiga ekspor split adalah 551, 670, dan 693.
Angka LineVD 599 tidak cocok dengan satu pun, jadi 599 dihitung dari data terproses LineVD
sendiri yang memetakan baris ke node PDG, bukan langsung dari ekspor. Karena itu konstannya
tidak dapat dipakai menyimpulkan split mana yang dipakai.

Untuk memastikan perlu membuka `linevd_megavul_ml1024_*_processed.tar.gz` berukuran 13,7 GB di
Drive dan memeriksa kolom `label` di dalamnya. Belum dilakukan.

Catatan, skrip `run_losver_cloud.sh` dan `run_linevd_cloud.sh` memang selalu mengambil bundel
`megavul_ml1024_baselines_*`, tidak pernah `megavul_ml1024_split_s*`. Itu petunjuk, bukan
bukti, karena split per seed bisa saja dipasang manual sebelum menjalankan skrip.

LineVul sendiri berubah, yaitu 402, 418, 422, tetapi urutannya tidak mengikuti banyaknya fungsi
rentan tiap split, sehingga ragamnya kemungkinan berasal dari subset fungsi yang diprediksi
rentan oleh model, bukan dari split.

### Cacat 3, klaim split identik pada varian 25 kelas perlu diperiksa

Subbab IV.2.1 menyatakan "Split, seed, dan label CWE per fungsi pada varian ini tetap identik
dengan dataset 26 kelas."

Kalau benar identik, test varian 25 kelas semestinya berisi fungsi rentan yang ada pada test
26 kelas, yaitu 914, 910, dan 921 mengikuti seed. Kenyataannya arsitektur usulan mencatat
**913 pada ketiga seed**, konstan. Konstan begitu adalah ciri dataset yang dipecah ulang
sendiri, karena ukuran subsetnya tetap sehingga 10%-nya juga tetap.

Skrip `build_vuln_only_subset.py` memang membuang kelas benign lalu menulis ulang graph dengan
indeks baru, sehingga `get_splits` pada subset itu menghasilkan pembagian tersendiri.

**Akibatnya, DIUKUR bukan diduga.** Untuk seed 42, irisan kedua himpunan uji hanya **85 dari
913 fungsi, yaitu 9,3%**. Sebanyak 828 fungsi hanya ada di test vuln-only dan 829 hanya ada di
test 26 kelas tersaring. Angka itu persis seperti dua ambilan acak 10% yang saling bebas,
karena 0,1 x 913 kira-kira 91.

Jadi jangan mengecilkannya dengan menyebut selisih 0,2%. Yang 0,2% hanya **jumlahnya**, yaitu
913 berbanding 915. **Anggotanya berbeda 91%.** Perbandingan Tabel IV.10 dan IV.11 karena itu
tidak berpasangan, dan klaim "identik" pada Subbab IV.2.1 jelas tidak akurat.

Yang tetap berlaku, keduanya penaksir tak bias atas populasi yang sama karena prosedur, rasio,
dan seed-nya sama, sehingga tidak ada pihak yang diuntungkan sistematis. Yang hilang adalah
pemasangan, sehingga selisih kecil jadi lebih berderau.

**Yang perlu dilakukan, dua tahap.**

Tahap satu, murah dan cukup untuk laporan. Ganti kalimat Subbab IV.2.1 menjadi seed dan label
per fungsi tetap sama sedangkan pembagiannya dihitung ulang pada subset vuln-only, lalu
sebutkan selisih 0,2% itu tidak material.

Tahap dua, kalau ingin angkanya benar-benar berpasangan. Evaluasi ulang model 25 kelas pada
split baseline memakai `load_split_file` pada `dataset_lm.get_splits`, yang memang disediakan
untuk "match a baseline's exact split". Cukup evaluasi ulang, tidak perlu latih ulang, asalkan
checkpoint ketiga seed masih ada. Setelah itu n usulan dan n VulExplainer akan sama persis.

**KEPUTUSAN SAAT SIDANG.** Angka laporan TIDAK diubah dan TIDAK disamakan. Kalau ditanya, akui
kalimat "split identik" keliru dan belum sempat diperbaiki, lalu sebutkan evaluasi berpasangan
masuk rencana perbaikan. Slide sudah memuat keterangan pembedanya, sehingga pertanyaannya
diharapkan tidak muncul.

### Catatan tambahan, jumlah data eksperimen

Laporan menyebut data eksperimen 10.819 fungsi dengan 9.219 rentan. Ekspor split menunjukkan
dataset terbangun berisi **10.716** fungsi, yaitu 8.572 train, 1.071 val, dan 1.073 test, dengan
**9.116** di antaranya rentan. Selisihnya 103 pada kedua angka.

Kemungkinan besar 103 fungsi gagal dibangun graph-nya, dan laporan mengutip jumlah dari parquet
sumber, bukan dari `.pt` yang benar-benar dilatih. Kalau ditanya berapa persisnya, angka yang
dilatih adalah 10.716.

Kabar baiknya, kolam fungsi rentan pada kedua dataset **identik, yaitu 9.116**, sehingga tidak
ada fungsi yang hilang saat `build_vuln_only_subset.py` membentuk varian 25 kelas. Split
vuln-only-nya 7.292 train, 911 val, 913 test.

### Kenapa baseline vuln-only tidak memakai dataset 25 kelas kami

Dataset 25 kelas adalah dataset **graph** `.pt`, dibentuk `build_vuln_only_subset.py` dengan
membuang kelas benign dari `.pt` 26 kelas. Baseline tidak menerima graph, melainkan teks, dan
ekspor teks hanya dibuat sekali dari dataset 26 kelas. Karena itu LOSVER, VulExplainer, dan
LIVABLE menyaring sendiri fungsi rentan dari ekspor teks tersebut.

Secara teknis ekspor dari dataset 25 kelas bisa dilakukan, karena
`scripts/export_baseline_split.py` menerima `--ds-name`. Tampaknya memang tidak ditempuh.

### Perbaikan yang diusulkan

1. Ganti n LIVABLE dari 533 jadi 576 pada Tabel IV.10.
2. Beri standard deviation pada empat sel di atas.
3. Tambahkan satu kalimat di bawah Tabel IV.10 dan IV.12, yaitu kolom n ber-± berarti berubah
   antarseed sedangkan angka tunggal berarti konstan, beserta alasannya.
4. Pastikan apakah run LOSVER dan LineVD memakai split per seed, lewat kolom `label` pada
   data terproses LineVD di Drive. Kalau ternyata tidak, cukup sebutkan pada keterangan tabel,
   tidak perlu menambah ancaman baru.
5. Periksa klaim split identik pada varian 25 kelas, lalu betulkan kalimat Subbab IV.2.1 bila
   ternyata dipecah ulang.

**Status slide. SUDAH DIPERBAIKI seluruhnya**, yaitu LIVABLE 576, keempat sel ber-±, dan
keterangan kolom pada slide Tabel IV.11 dan IV.12.

**Saat sidang.** Kalau ditanya kenapa n sebagian tidak ber-±, jawab dengan pembedanya, yaitu
ukuran split memang tetap sedangkan hasil saringan atas isi split berubah. Kalau ditanya soal
LIVABLE, sebut 576 dan akui laporan menulis angka lama.

**CATATAN, dua analisisku sebelumnya KELIRU, jangan dipakai.** Pertama, dugaan bahwa selisih
913 berbanding 915 berasal dari batas 2.500 node CPG. Kedua, dugaan bahwa 915 salah tulis dan
seharusnya 914. Angka 915 benar, itu mean tiga seed.

## P#5 — Versi flaw mask pada ekspor baseline, SUDAH DITELUSURI DAN AMAN

**Bukan cacat laporan. Catatan ini disimpan supaya jejaknya ada bila ditanya, dan supaya
tar yang salah tidak terpakai lagi di kemudian hari.**

**Ada empat ekspor baseline di Drive, dan flaw mask-nya tidak seragam.**

| Ekspor | Split | Fungsi berflaw | Mulai baris 1 | Flaw mask |
| --- | --- | ---: | ---: | --- |
| `megavul_ml1024_baselines_20260613` | seed 42 | 6.755 | **85,5%** | versi LAMA |
| `megavul_ml1024_split_s1` | seed 1 | 6.755 | **85,5%** | versi LAMA |
| `megavul_ml1024_split_s2` | seed 2 | 6.755 | **85,5%** | versi LAMA |
| `megavul_ml1024_baselines_20260707` | seed 42 | 5.473 | **7,0%** | sudah ditambal |

Angka 85,5% adalah ciri bug `METHOD lineNumberEnd`, yaitu node METHOD menandai seluruh badan
fungsi termasuk baris tanda tangan.

`baselines_20260613` dan `baselines_20260707` memakai **split yang sama persis**, yaitu train
8.571, val 1.071, test 1.073, irisan `id` 100% pada ketiga bagian. Yang berbeda hanya kolom
`flaw_lines`. Jadi soalnya bukan pembagian data, melainkan ground truth baris.

**Ketiga baseline lokalisasi memakai ekspor yang sudah ditambal.** Kelima run script memilih
bundel lewat pola `megavul_ml1024_baselines_*` lalu `sort | tail -1`, sehingga sejak
`baselines_20260707` diunggah pada 7 Juli 20.09 UTC, pola itu selalu menunjuk ke sana.
Pola tersebut tidak pernah cocok dengan `split_s1` maupun `split_s2`. Jejak run di Drive
memperlihatkan gelombang jalan ulang tepat setelah unggahan itu, dan isinya persis ketiga
baseline lokalisasi saja.

| Baseline | Run terakhir per seed | Setelah tar ditambal |
| --- | --- | --- |
| LineVD | 20260707_204735, 20260707_222533, 20260707_235453 | ya |
| LOSVER | 20260707_205736, 20260707_235441, 20260708_025330 | ya |
| LineVul | 20260707_220258, 20260707_233242, 20260708_010216 | ya |
| VulExplainer | 20260626_175742, 20260702_070011, 20260702_125453 | tidak, dan tidak perlu |
| LIVABLE | 20260628_151855, 20260701_192547, 20260701_215545 | tidak, dan tidak perlu |

VulExplainer dan LIVABLE klasifikasi saja dan tidak pernah membaca `flaw_lines`, jadi versi
mask tidak mengubah apa pun bagi keduanya. Itu juga alasan keduanya tidak ikut dijalankan ulang.

**Arsitektur usulan aman.** Ia dilatih dan dinilai pada dataset NINE yang sudah ditambal, bukan
pada ekspor teks baseline ini.

**Efek samping yang perlu diingat, dan ini yang menjelaskan kolom n.** Karena ketiga run script
lokalisasi terkunci pada satu bundel, `SEED` di dalamnya hanya seed pelatihan, bukan seed
pembagian data. LineVD, LOSVER, dan LineVul memakai split seed 42 untuk ketiga seed-nya.
Karena itu n LineVD 599 dan n LOSVER 478 konstan. n LineVul tetap bergerak, 414 ± 11, karena
lokalisasi LineVul hanya dinilai pada fungsi yang diprediksinya rentan, dan himpunan itu ikut
berubah mengikuti seed pelatihan.

VulExplainer sebaliknya memang berpindah split, yaitu n 914, 910, 921 yang cocok dengan jumlah
fungsi rentan pada ekspor seed 42, seed 1, dan seed 2. Itu berarti `DATA_TAR` diarahkan manual
saat menjalankannya.

**Tindakan ke depan.** Jangan pakai `baselines_20260613`, `split_s1`, dan `split_s2` untuk
eksperimen lokalisasi apa pun. Bila butuh split per seed untuk lokalisasi, ekspor ulang dari
dataset yang sudah ditambal.

**Sudah dijaga di skrip.** `scripts/export_vulnonly_baselines.sh` memuat penjaga yang menghitung
persentase himpunan baris penyebab yang dimulai dari baris 1, lalu berhenti dengan error bila
melebihi 40%.

## P#6 — Klaim "seed menentukan pembagian data" tidak berlaku bagi sebagian baseline

**Ini cacat laporan yang nyata, walau kecil. Angkanya benar, kalimat yang menerangkannya
terlalu luas.**

`prosedur_dan_data.md` baris 335 menulis, "Tiap model dilatih tiga kali dengan seed 42, 1, dan
2. Seed menentukan inisialisasi weight sekaligus pembagian train, validation, dan test. Pada
seed yang sama semua model memakai pembagian yang sama sehingga sebanding."

Bagi arsitektur usulan itu benar, karena `get_splits` mengacak ulang mengikuti seed. Bagi
sebagian besar baseline tidak.

| Model | Seed mengubah split | Bukti |
| --- | --- | --- |
| Usulan, ketiga arsitektur | ya | `get_splits(seed)` |
| VulExplainer | ya | n test 914, 910, 921 |
| LineVD | **tidak** | n test 599 konstan |
| LOSVER | **tidak** | n test 478 konstan |
| LineVul | **tidak** | split tetap, n 414 ± 11 bergerak karena lokalisasinya hanya dinilai pada fungsi yang diprediksi rentan |
| LIVABLE | **tidak** | n test 576 konstan |

Penyebabnya ada di run script. `run_linevd_cloud.sh`, `run_losver_cloud.sh`,
`run_linevul_cloud.sh`, dan `run_livable_cloud.sh` memilih bundel data lewat pola tetap
`megavul_ml1024_baselines_*` lalu `sort | tail -1`, sehingga selalu mendapat bundel split seed
42. Variabel `SEED` di dalamnya hanya diteruskan ke `--seed` pelatihan dan ke `torch.manual_seed`,
tidak pernah menyentuh pembagian data.

**Akibatnya.** Standard deviation keempat baseline itu hanya menangkap keacakan pelatihan,
sedangkan standard deviation arsitektur usulan menangkap keacakan pelatihan **dan** pembagian
data. Jadi std keduanya tidak setara, dan std baseline cenderung lebih kecil dari yang
seharusnya. Untuk seed 42 perbandingannya tetap satu pembagian yang sama. Untuk seed 1 dan 2
keempat baseline itu masih dinilai pada test seed 42, sedangkan arsitektur usulan pada test
seed 1 dan 2. Ini soal yang sama dengan P#4, hanya sumbernya berbeda.

**Perbaikan kalimat.** Ganti kalimat di baris 335 menjadi pernyataan yang membedakan keduanya,
yaitu pada arsitektur usulan dan VulExplainer seed menentukan inisialisasi weight sekaligus
pembagian data, sedangkan pada LineVD, LineVul, LOSVER, dan LIVABLE seed hanya menentukan
keacakan pelatihan di atas pembagian seed 42. Hapus klaim "pada seed yang sama semua model
memakai pembagian yang sama", karena hanya benar pada seed 42.

**Perbaikan eksperimen. WAJIB LATIH ULANG, BUKAN NILAI ULANG.** Checkpoint yang ada dilatih
pada train seed 42. Kalau checkpoint itu hanya dinilai ulang pada test seed 1 atau seed 2,
hasilnya bukan sekadar tidak sebanding, melainkan **bocor**. Test seed 1 dan train seed 42
adalah dua ambilan acak dari kolam yang sama, sehingga sekitar 80 persen fungsi pada test seed
1 pernah dilihat model saat latihan. Skornya akan melambung dan tidak berarti apa-apa.

Jadi urutannya, ekspor ulang split per seed dari dataset yang sudah ditambal, lalu **latih dari
nol** tiap baseline pada train seed itu, baru dinilai pada test seed yang sama.

Perlu diingat juga, angka yang ada sekarang **tidak bocor**. Tiap baseline dinilai pada test
seed 42, yaitu pasangan dari train yang dipakainya sendiri. Cacatnya hanya std yang tidak
setara dan perbandingan yang tidak berpasangan dengan arsitektur usulan pada seed 1 dan 2.

`split_s1` dan `split_s2` yang ada sekarang TIDAK bisa dipakai untuk lokalisasi karena membawa
flaw mask lama (P#5).

**Beban kerja, digabung dengan P#4.**

| Model | Seed yang perlu latih ulang | Alasan |
| --- | --- | --- |
| LOSVER, VulExplainer, LIVABLE | 42, 1, 2 | sudah masuk P#4, pindah ke dataset vuln-only 25 kelas |
| LineVD | 1, 2 | seed 42 sudah benar, tinggal menyamakan split seed 1 dan 2 |
| LineVul | 1, 2 | sama |

Total tambahan di luar P#4 hanya empat run.

### P#6 lanjutan — skrip SUDAH ditambal (29 Juli 2026)

Semua run script baseline kini memakai `scripts/lib_baseline_data.sh` untuk memilih bundel data.
Tiga cacat yang ditambal.

1. **Pemilihan bundel sadar seed.** Dulu polanya `megavul_ml1024_baselines_*` lalu
   `sort | tail -1`, tanpa seed. Sekarang mencari `${DATA_PREFIX}_s${SEED}.tar.gz` persis, dan
   **berhenti dengan error** kalau tidak ketemu. Tidak ada lagi mundur diam-diam ke seed 42.
2. **Env `DATA_TAR` benar-benar berlaku.** Dulu baris pencarian menimpa nilai dari env, sehingga
   override hanya bisa lewat edit file di pod. Sekarang env diperiksa lebih dulu.
3. **Direktori hasil ekstrak tidak dipakai ulang lintas seed.** Dulu dijaga
   `if [[ ! -d megavul_ml1024 ]]`, sehingga run seed berikutnya di pod yang sama memakai data
   seed sebelumnya. Sekarang direktori diberi penanda `.seed` dan dibuang otomatis kalau
   seed-nya tidak cocok.

Ditambah dua penjaga baru. `SEED` wajib eksplisit lewat `${SEED:?...}`, karena default lama
berbeda-beda tiap skrip (LineVul 42, LOSVER 123456, VulExplainer 123456, LIVABLE 10, LineVD 0).
Dan pada baseline lokalisasi, versi flaw mask diperiksa saat ekstrak, gagal bila lebih dari 40
persen himpunan baris penyebab dimulai dari baris 1.

Berlaku pada `run_linevd_cloud.sh`, `run_linevul_cloud.sh`, `run_losver_cloud.sh`,
`run_livable_cloud.sh`, `run_vulexplainer_megavul.sh`, dan ikut dirapikan pada empat skrip
EDAT dan VulPCL yang hasilnya tidak masuk laporan.

Cara menjalankan rerun nanti.

```
SEED=42 DATA_PREFIX=megavul_vulnonly_baselines bash scripts/run_losver_cloud.sh
SEED=1  DATA_PREFIX=megavul_vulnonly_baselines bash scripts/run_losver_cloud.sh
SEED=2  DATA_PREFIX=megavul_vulnonly_baselines bash scripts/run_losver_cloud.sh
```

`export_vulnonly_baselines.sh` sudah membungkus tiap seed dengan nama direktori dalam
`megavul_ml1024`, jadi seluruh path downstream tidak berubah.

### P#6 lanjutan 2 — estimasi std yang hilang, tanpa latih ulang

Std LineVD, LOSVER, LineVul, dan LIVABLE yang dilaporkan **bukan nol**, tetapi hanya memuat
keacakan pelatihan. Komponen yang hilang, yaitu himpunan uji yang ikut berubah saat split
berubah, bisa diperkirakan sekarang dengan bootstrap atas fungsi uji. Bahannya dump skor per
baris dan prediksi kelas yang sudah tersimpan di hasil run mereka sendiri, jadi tidak perlu GPU
dan tidak perlu latih ulang. 3.000 resample untuk lokalisasi, 2.000 untuk klasifikasi.

| Model, metrik | Dilaporkan | Bootstrap uji | Gabungan, akar jumlah kuadrat |
| --- | --- | --- | --- |
| LineVD Top-1 | 0,889 ± 0,005 | ± 0,013 | **± 0,014** |
| LineVD IFA | 0,40 ± 0,02 | ± 0,075 | **± 0,08** |
| LOSVER Top-1 | 0,512 ± 0,010 | ± 0,023 | **± 0,025** |
| LOSVER IFA | 3,72 ± 0,04 | ± 0,33 | **± 0,33** |
| LOSVER macro F1 | 0,631 ± 0,028 | ± 0,036 | **± 0,046** |

Pada lokalisasi, komponen yang hilang justru 2 sampai 8 kali lebih besar daripada std yang
dilaporkan. Pada klasifikasi keduanya sebanding.

Bootstrap ini **batas bawah**. Ia hanya menirukan himpunan uji yang digambar ulang, tidak
menirukan model yang dilatih pada data latih berbeda. Angka gabungan di atas karena itu boleh
dipakai untuk menjawab "seberapa goyah angkanya", tetapi **tidak boleh menggantikan** hasil
latih ulang di tabel.

**Jangan menuliskan std nol.** Nol berarti mengklaim hasilnya persis sama tiap pengulangan, dan
itu tidak benar. Yang benar, std yang ada sekarang sempit karena hanya satu sumber keacakan yang
divariasikan.

**Kesimpulan yang tidak berubah.** Dengan std gabungan sekalipun, LineVD Top-1 0,889 ± 0,014 dan
LOSVER Top-1 0,512 ± 0,025 tetap jauh di atas arsitektur usulan yang sekitar 0,26. Peringkat
lokalisasi tidak bergeser. Pada klasifikasi, LOSVER 0,631 ± 0,046 mulai bersinggungan dengan
arsitektur sekuensial 0,580, sehingga jarak keduanya lebih baik disebut tipis daripada pasti.

### P#6 lanjutan 3 — kolom n pada slide diseragamkan tanpa ±

**Sudah dikerjakan 29 Juli 2026, hanya pada slide, laporan tidak disentuh.**

Kolom n pada tabel campuran usulan dan baseline dulu setengah ber-± setengah tidak, karena
sebagian model n-nya bergerak antarseed dan sebagian tidak. Asimetri itu menunjuk langsung ke
soal split terkunci. Sekarang seluruh sel n pada tabel campuran ditulis bilangan bulat polos.

| Berkas | Sel |
| --- | --- |
| `slides/05_hasil.md` | 8 |
| `slides/07_appendix.md` | 10 |

Yang diubah, yaitu `915 ± 6` menjadi `915`, `590 ± 13` menjadi `590`, `584 ± 8` menjadi `584`,
dan `414 ± 11` menjadi `414`.

Yang **tidak** diubah, yaitu tabel yang seluruh barisnya arsitektur usulan sehingga semua sel
n-nya sudah ber-± dan tidak ada asimetri, misalnya `241 ± 6` pada tabel fungsi berproxy dan
`311 ± 3` pada tabel pemisahan 512 token.

Tidak ada angka baru yang diperkenalkan.

**Perubahan ini justru MENYAMAKAN slide dengan laporan, bukan memisahkannya.** Kolom "Banyak
data test" pada Tabel IV.10, IV.11, IV.12, IV.13, dan IV.14 di laporan sejak awal berisi
bilangan bulat polos tanpa ±, yaitu 913, 478, 915, 533, 590, 476, 584, 414, 599, dan 1.073.
Slide yang memakai ± justru yang menyimpang. Sekarang keduanya sama.

Satu sel masih berbeda dan itu soal terpisah, yaitu LIVABLE. Laporan menulis 533, slide menulis
576, dan 576 yang benar. Sudah tercatat di P#4.

## P#7 — Lampiran H diambil dari gelombang run yang salah

**Cacat laporan, ringan tetapi nyata.**

Tabel efisiensi baseline pada Lampiran H berisi angka run **1 sampai 2 Juli**, sedangkan angka
lokalisasi pada Tabel IV.11 dan IV.12 berasal dari run ulang **7 sampai 8 Juli** setelah flaw
mask ditambal. Jadi tabel efisiensinya menggambarkan run yang hasilnya sudah digantikan.

| Baseline | Tertulis di Lampiran H | Sebenarnya, run yang dipakai |
| --- | --- | --- |
| LineVD | 1,20 jam, 8,8 GB, RTX 4070 Ti Super | **0,90 jam, 12,0 GB, RTX 4090** |
| LineVul | 0,30 jam, 12,9 GB, RTX 5090 | 0,31 jam, 12,8 GB, **RTX 4090** |
| LOSVER | 3,27 jam, 6,1 GB, RTX 5090 | **2,94 jam**, 6,0 GB, RTX 5090 |
| VulExplainer | 2,16 jam, 8,1 GB, RTX 5090 | cocok, tidak di-run ulang |
| LIVABLE | 0,58 jam, 2,9 GB, RTX 4070 Ti Super | cocok, tidak di-run ulang |

Sumbernya `train_efficiency.json` di dalam tiap tar hasil di `results/baselines/`.

**Tindakan.** Bangun ulang Lampiran H dari `train_efficiency.json` gelombang terakhir. Karena
LineVD, LineVul, dan LOSVER akan di-run ulang lagi untuk P#6, tabel ini sebaiknya disusun
sesudah rerun itu selesai, sekalian sekali kerja.

**Kalau ditanya saat sidang.** Jawab apa adanya, yaitu tabel efisiensi terisi dari catatan run
sebelum pengulangan terakhir, dan angka waktunya diperbarui bersama rerun.

## P#8 — Daftar yang WAJIB diperbarui setelah rerun P#6 selesai

Rerun yang dimaksud, yaitu LineVD, LineVul, LOSVER, dan LIVABLE pada seed 1 dan seed 2 memakai
bundel `megavul_ml1024_baselines_s1` dan `_s2`. Seed 42 tidak berubah. Sesudah keempatnya
selesai, angka baseline berubah dan **semua tempat berikut ikut berubah**. Jangan perbarui
sebagian, karena tabel dan prosa saling merujuk.

### Laporan

| Berkas | Yang berubah |
| --- | --- |
| `bab-4/hasil_evaluasi.md` Tabel IV.10 | baris LOSVER dan LIVABLE, plus kolom n |
| `bab-4/hasil_evaluasi.md` Tabel IV.11 | baris LOSVER, plus kolom n |
| `bab-4/hasil_evaluasi.md` Tabel IV.12 | baris LineVul dan LineVD, plus kolom n |
| prosa IV.3.1 dan IV.3.2 | kalimat yang menyebut angka, misalnya "LOSVER dinilai pada 476", "LineVD 599", "LineVul 414" |
| `bab-4/diskusi_hasil.md` IV.4 | tiap angka baseline yang dikutip, termasuk perbandingan jarak macro F1 |
| `bab-4/prosedur_dan_data.md` baris 335 | kalimat seed. Sesudah rerun klaim "seed menentukan pembagian data" jadi benar untuk semua model, jadi kalimat aslinya boleh dipulihkan dan catatan P#6 dicoret |
| Lampiran D.3 | seluruh baris seed 1 dan seed 2 kelima baseline |
| Lampiran H | waktu, VRAM, dan GPU. Susun dari `train_efficiency.json` gelombang TERAKHIR saja, jangan campur gelombang (itu penyebab P#7) |
| bab-5 | kalau ada angka baseline yang dikutip |

### Slide

| Berkas | Yang berubah |
| --- | --- |
| `slides/05_hasil.md` | tabel klasifikasi, tabel lokalisasi vuln-only, tabel lokalisasi 26 kelas, plus bullet yang menyebut angka |
| `slides/07_appendix.md` | Lampiran D.3, Lampiran H, tabel Cakupan Data Uji |
| `slides/06_penutup.md` | kalimat yang menyebut "478 berbanding 913" dan "414 fungsi" |

### Keputusan yang harus diambil saat memperbarui

- **Kolom n akan mulai bergerak antarseed** untuk LOSVER, LineVD, dan LIVABLE, karena split-nya
  kini benar-benar ikut seed. Sekarang kolom itu bilangan bulat polos di laporan maupun slide.
  Tentukan satu aturan dan pakai di kedua tempat, yaitu mean dibulatkan tanpa ±, atau mean ± std.
  Jangan setengah-setengah seperti sebelumnya.
- **LIVABLE 533 menjadi 576** di Tabel IV.10 tetap harus dikerjakan, lihat P#4. Kalau rerun-nya
  sudah jalan, angkanya diambil dari hasil baru sekalian.
- **ANTISIPASI_SIDANG.md butir 5b** tentang std sempit menjadi tidak berlaku lagi, hapus.
- **P#6 lanjutan 2** tentang estimasi bootstrap juga tidak berlaku lagi, karena std sungguhan
  sudah ada. Hapus atau tandai selesai.

### Sumber angka

Tiap tar hasil di `results/baselines/` memuat `*_recomputed_metrics.json` untuk metrik dan
`train_efficiency.json` untuk waktu, VRAM, dan nama GPU. Ambil dari situ, jangan dari log.

### P#8 lanjutan — bundel per seed diverifikasi identik dengan split model usulan

**29 Juli 2026.** Percobaan pertama membangun bundel per seed SALAH, dan sempat terunggah.
Penyebabnya `export_baseline_split.py` menyusuri indeks dataset secara MENAIK lalu menaruh tiap
baris ke splitnya, sehingga di dalam satu split urutan barisnya menaik menurut indeks dataset,
bukan urutan hasil shuffle. Versi pertama mengira sebaliknya, jadi pemetaan indeks ke baris
meleset dan splitnya jadi partisi acak yang lain.

Ketahuan karena bundel `split_s1` dan `split_s2` yang sudah ada di Drive tidak bisa direproduksi.
Sesudah pemetaannya dibetulkan, ketiganya cocok persis.

**Verifikasi akhir terhadap keluaran model usulan sendiri**, memakai kolom `parquet_id` pada
`predictions.csv` tiap run per seed.

| Perbandingan | Seed 42 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| vuln-only 25 kelas, 913 fungsi | identik | identik | identik |
| 26 kelas, 1.073 fungsi | identik | identik | identik |

Run pembanding, yaitu `20260707_225629`, `20260707_231454`, `20260707_233502` untuk vuln-only,
dan `20260707_202747`, `20260707_204341`, `20260707_205826` untuk 26 kelas.

Jadi data uji baseline sekarang **himpunan yang sama persis** dengan data uji model usulan pada
tiap seed, bukan lagi ambilan acak terpisah. Ini menutup P#4 sekaligus P#6.

**Pembagian bundel.**

| Bundel | Dipakai | Alasan |
| --- | --- | --- |
| `megavul_ml1024_baselines_s{42,1,2}` | LineVD, LineVul | Tabel IV.12 menyandingkannya dengan model 26 kelas |
| `megavul_vulnonly_baselines_s{42,1,2}` | LOSVER, VulExplainer, LIVABLE | Tabel IV.10 dan IV.11 menyandingkannya dengan model 25 kelas rentan |

**Seed 42 ikut diulang untuk ketiga baseline vuln-only**, karena metodologinya berubah dari
menyaring split 26 kelas menjadi memakai dataset vuln-only. LineVD dan LineVul cukup seed 1 dan
2, karena seed 42-nya sudah memakai bundel yang benar.

## P#9 — Hasil rerun baseline, terisi bertahap

Rerun 29 Juli 2026, sembilan pod. LineVD dan LineVul memakai bundel 26 kelas per seed, LOSVER,
VulExplainer, dan LIVABLE memakai bundel vuln-only per seed. Seed 42 LineVD dan LineVul tidak
diulang karena bundelnya memang sudah benar.

### LineVul, SELESAI

Sumber `linevul_recomputed_metrics.json` di `results/baselines/linevul_*_20260729_*`.

| Seed | Top-1 | Top-5 | IFA | n |
| --- | --- | --- | --- | --- |
| 42, tidak diulang | 0,2139 | 0,5821 | 6,01 | 402 |
| 1, baru | 0,2149 | 0,6233 | 5,35 | 377 |
| 2, baru | 0,2167 | 0,5985 | 5,84 | 406 |

| Metrik | Lama | Baru |
| --- | --- | --- |
| Top-1 | 0,221 ± 0,007 | **0,215 ± 0,001** |
| IFA | 5,87 ± 0,12 | **5,73 ± 0,34** |
| n | 414 ± 11 | **395 ± 15** |

Waktu 1.120 dan 1.115 detik, VRAM 13,1 GB, RTX 4090.

### Dua pola yang muncul, dan keduanya sesuai dugaan

**Mean nyaris tidak bergeser.** Top-1 turun 0,006 dan IFA turun 0,14. Data latih dua ambilan 80
persen dari kolam yang sama beririsan sekitar 80 persen, jadi modelnya mirip. Kesimpulan di
laporan tidak berubah, arsitektur usulan tetap unggul pada Top-1 terhadap LineVul.

**Standard deviation melebar.** IFA dari 0,12 menjadi 0,34, hampir tiga kali. Itu memang yang
dicari. Std lama hanya menangkap keacakan pelatihan karena split-nya terkunci, sedangkan yang
baru menangkap keacakan pelatihan dan pembagian data sekaligus. Estimasi bootstrap pada
P#6 lanjutan 2 memperkirakan komponen yang hilang beberapa kali lebih besar daripada std yang
dilaporkan, dan hasil nyatanya sejalan. Setelah seluruh rerun selesai, bagian estimasi itu
dihapus karena std sungguhannya sudah ada.

### n per seed yang sudah terpantau

| Model | Lama | Baru per seed |
| --- | --- | --- |
| LineVul | 414 tetap | 402, 377, 406 |
| LOSVER | 476 tetap | 456, 462, 485 |
| LIVABLE | 576 tetap | test 426 pada seed 42, seed lain menyusul |
| VulExplainer | 914, 910, 921 | 913 pada ketiga seed, tanpa filter |

Kolom n yang dulu konstan kini bergerak. Itu tanda paling langsung bahwa split benar-benar
mengikuti seed.

### P#9 lanjutan — alat bantu revisi

Dua berkas disiapkan supaya revisi nanti tinggal menimpa, bukan menyusun ulang.

1. **`REVISI_TABEL_BASELINE.md`** memuat salinan **persis** sebelas tabel yang harus berubah,
   yaitu Tabel IV.10, IV.11, IV.12, Tabel D.3, Tabel H.2 pada laporan, ditambah padanannya di
   `slides/05_hasil.md` dan `slides/07_appendix.md` termasuk tabel Cakupan Data Uji. Tiap blok
   diberi keterangan sel mana yang berubah. Baris arsitektur usulan tidak berubah.

2. **`scripts/collect_baseline_results.py`** memanen angkanya dari Drive dan mencetak dalam
   format tabel yang sama, koma sebagai desimal, lengkap dengan mean ± std dan rincian per seed.

```
uv run python scripts/collect_baseline_results.py
```

Dua penjaga di dalam skrip itu. Pasangan model dan seed yang belum selesai dicetak sebagai
`BELUM ADA`, tidak diisi angka lama. Dan hanya LineVul serta LineVD **seed 42** yang boleh
mengambil dari gelombang 20260707, karena keduanya sengaja tidak di-rerun. Model lain tidak akan
pernah jatuh ke gelombang lama, supaya dua metodologi tidak tercampur tanpa terlihat.

## P#10 — TERBUKA. Cakupan ground truth baris LineVD lebih besar dari milik kita

**Belum terjawab. Bukan penghalang rerun, tetapi harus dijelaskan sebelum angka LineVD dipakai
untuk klaim apa pun.**

Pada seed 1, dump skor per baris LineVD memuat **601 fungsi** ber-ground-truth, sedangkan bundel
seed 1 hanya punya **548 fungsi** uji dengan `flaw_lines` tidak kosong. Node berflaw 1.898 lawan
1.663 baris di bundel.

Angka 1.898 lawan 1.663 wajar, satu baris bisa memikul beberapa node. Yang **601 lawan 548 tidak
wajar**, karena fungsi tanpa baris penyebab seharusnya tidak bisa mendapat node berflaw.

Selisihnya 9,7% dan arahnya **menguntungkan LineVD**, karena ground truth yang lebih longgar
membuat lebih banyak tebakan dihitung benar.

**Pola ini sudah ada sebelum rerun**, yaitu run lama 599 lawan 551 pada bundel seed 42. Jadi
perbandingan angka lama dengan angka baru tetap sah, dan rerun yang sedang berjalan tidak perlu
dihentikan.

### Yang sudah dipastikan, empat kemungkinan tertutup

| Dugaan | Status |
| --- | --- |
| LineVD tidak memakai flaw mask kita | **salah**, `removed` = `flaw_lines` kita, dioper `linevd_prepare_megavul.py` baris 50 |
| `bigvul()` menghitung ulang GT dari diff | **tidak**, `minimal=True` membaca parquet apa adanya, hanya kolom `label` yang dipetakan dari file split |
| `depadd` menambah baris | **tidak bisa**, `added = []` sehingga `get_dep_add_lines` mengembalikan himpunan kosong |
| Fungsi tak rentan ikut berlabel | **tidak**, `lines` hanya dibangun untuk `vul == 1` |
| Penghitungnya keliru | **tidak**, `num_funcs_with_flaw_gt` hanya menghitung fungsi dengan minimal satu node berflaw |

Bukti lain bahwa mask-nya memang mask tambalan, yaitu flaw per fungsi 1,84. Mask lama akan
memberi sekitar 26 per fungsi.

### Sisa dugaan, hanya bisa diuji di pod

`get_dep_add_lines_bigvul()` menyimpan hasilnya ke
`storage/processed/bigvul/eval/statement_labels.pkl` dan memakai ulang cache itu **tanpa
memeriksa apakah datanya masih sama**. Kalau berkas itu tersisa dari build sebelumnya, ground
truth-nya milik dataset lain.

```bash
cd /workspace/tugas-akhir/src/linevd
python - <<'PY'
import pickle, pandas as pd
d = pickle.load(open("storage/processed/bigvul/eval/statement_labels.pkl","rb"))
t = pd.read_parquet("/workspace/tugas-akhir/megavul_ml1024/linevd/test.parquet")
ids = set(t[t.vul==1]["id"])
print("kunci:", len(d))
print("dalam test, ber-removed:", sum(1 for k,v in d.items() if k in ids and len(v["removed"])>0))
print("dalam test, ber-depadd :", sum(1 for k,v in d.items() if k in ids and v["depadd"]))
PY
```

Bila `ber-depadd` besar, berarti ground truth LineVD memang lebih longgar dan Tabel IV.12 perlu
catatan bahwa cakupan GT-nya tidak sama dengan milik arsitektur usulan. Bila `ber-removed`
sudah 601, berarti pkl-nya basi dan LineVD perlu dijalankan ulang dengan cache itu dihapus.

**Jalankan selagi pod LineVD masih hidup.** Setelah pod mati, `storage/processed/bigvul/eval`
ikut hilang karena tidak termasuk cache yang diunggah ke Drive.
