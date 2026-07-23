"""Pengujian fungsional kontrak modul (Subbab III.2.6).

Tiap test memetakan satu janji rancangan modul, bukan satu fungsi internal:

  III.2.6.2  konfigurasi sebagai antarmuka   -> T1, T2
  III.2.6.3  tiga kelompok kemampuan         -> T3, T4, T7, T8
  III.2.6.4  identitas artefak & ruang kelas -> T5, T6

Tier cepat jalan tanpa Joern, tanpa unduhan LM, dan tanpa GPU. Tier lambat ditandai
`@pytest.mark.slow` karena memanggil Joern dan melatih model kecil.

    uv run pytest tests/test_kontrak_modul.py -v -m "not slow"
    uv run pytest tests/test_kontrak_modul.py -v            # termasuk tier lambat
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

# gnn_vuln.core memuat pandas dan pyarrow, dan pada Windows pyarrow harus dimuat sebelum
# torch. Urutan sebaliknya membuat proses mati dengan access violation.
from gnn_vuln.core import check_compatible

import torch

from gnn_vuln.config import Config
from gnn_vuln.models.registry import build_model
from gnn_vuln.research.naming import derive_ds_name

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs" / "ablation" / "gnn_only" / "N48_a1_l1_jknet.yaml"
HYBRID_CONFIG = REPO / "configs" / "ablation" / "phase13" / "O1_unixcoder_nine.yaml"
JOERN = os.environ.get("GNN_VULN_JOERN_CLI", "C:/joern/joern-cli")

FEATURISASI = dict(
    source="megavul", mode="multiclass", lm_short="unixcoder-base-nine",
    func_short="unixcoder-base-nine", add_func_tokens=True, func_max_length=1024,
    top_cwe=0, fsuffix="_f40f2e964", max_per_class=1600, resample_seed=42,
)


def _config(architecture: str, num_classes: int = 6) -> Config:
    cfg = Config.from_yaml(str(BASE_CONFIG))
    cfg.model.architecture = architecture
    cfg.model.num_classes = num_classes
    cfg.model.hidden_dim = 32
    cfg.model.num_layers = 2
    cfg.model.heads = 2
    return cfg


def _batch(n_graphs: int = 2, n_nodes: int = 12, in_channels: int = 64, edge_dim: int = 31):
    """Satu batch sintetis berbentuk CPG kecil, cukup untuk menguji kontrak keluaran."""
    torch.manual_seed(0)
    x = torch.randn(n_nodes, in_channels)
    src = torch.arange(n_nodes - 1)
    edge_index = torch.stack([src, src + 1])
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    batch[n_nodes // n_graphs:] = 1
    node_line = torch.arange(n_nodes) % 5
    return dict(x=x, edge_index=edge_index, batch=batch, node_line=node_line,
                edge_attr=edge_attr)


# ── III.2.6.2 Konfigurasi sebagai antarmuka ───────────────────────────────────

@pytest.mark.parametrize("architecture", ["lmgat_codebert", "lmgat_seqgnn"])
def test_t1_arsitektur_ditukar_lewat_satu_nilai_konfigurasi(architecture):
    """T1 — pemanggil menukar arsitektur tanpa menyentuh kode pemanggilan."""
    cfg = _config(architecture)
    model = build_model(cfg, in_channels=64).eval()
    with torch.no_grad():
        out = model(**_batch())
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, cfg.model.num_classes)


@pytest.mark.slow
def test_t1b_arsitektur_hibrida_memakai_antarmuka_yang_sama():
    """T1b — hibrida dibangun dari konfigurasinya sendiri dan memenuhi kontrak yang sama.

    Dipisah dari T1 dan ditandai lambat karena hibrida menjalankan LM tingkat fungsi,
    sehingga menuntut weight UniXcoder, sedangkan dua arsitektur lain tidak.
    """
    cfg = Config.from_yaml(str(HYBRID_CONFIG))
    cfg.model.num_classes = 6
    cfg.model.num_layers = 2
    cfg.model.heads = 2
    cfg.model.func_max_length = 128
    cfg.model.func_chunk_size = 128

    model = build_model(cfg, in_channels=64).eval()
    batch = _batch()
    n_token = 128
    batch["func_input_ids"] = torch.randint(0, 1000, (2, n_token))
    batch["func_attention_mask"] = torch.ones(2, n_token, dtype=torch.long)
    batch["func_token_lines"] = torch.randint(0, 5, (2, n_token))

    with torch.no_grad():
        out = model(**batch)
    assert out[0].shape == (2, cfg.model.num_classes)
    assert isinstance(out[1], list) and len(out[1]) == 2


def test_t2_arsitektur_tak_dikenal_ditolak():
    """T2 — konfigurasi salah gagal saat dibangun, bukan diam-diam salah hitung."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model(_config("arsitektur_yang_tidak_ada"), in_channels=64)


# ── III.2.6.3 Kemampuan inferensi, kontrak keluaran ganda ─────────────────────

@pytest.mark.parametrize("architecture", ["lmgat_codebert", "lmgat_seqgnn"])
def test_t3_satu_masukan_menghasilkan_dua_keluaran(architecture):
    """T3 — klasifikasi tingkat fungsi dan skor per unit keluar dari satu panggilan."""
    cfg = _config(architecture)
    model = build_model(cfg, in_channels=64).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(**batch)
    assert isinstance(out, tuple) and len(out) >= 2
    skor_baris = out[1]
    assert isinstance(skor_baris, list) and len(skor_baris) == 2, "skor baris tidak per fungsi"
    for i, skor in enumerate(skor_baris):
        n_baris = batch["node_line"][batch["batch"] == i].unique().numel()
        assert skor.shape == (n_baris,), "satu skor kecurigaan per baris (Subbab III.2.2.3)"


def test_t4_tingkat_keyakinan_berupa_distribusi_probabilitas():
    """T4 — luaran Subbab I.3 menuntut tingkat keyakinan tiap kelas, bukan label saja."""
    cfg = _config("lmgat_codebert")
    model = build_model(cfg, in_channels=64).eval()
    with torch.no_grad():
        logits = model(**_batch())[0]
    prob = torch.softmax(logits, dim=-1)
    assert torch.allclose(prob.sum(dim=-1), torch.ones(prob.shape[0]), atol=1e-5)
    assert (prob >= 0).all()


# ── III.2.6.4 Identitas artefak ───────────────────────────────────────────────

def test_t5_identitas_artefak_mengikuti_featurisasi():
    """T5 — konfigurasi sama memberi nama sama, satu parameter berubah memberi nama beda."""
    nama = derive_ds_name(**FEATURISASI)
    assert derive_ds_name(**FEATURISASI) == nama

    for ubah in ({"func_max_length": 5120}, {"lm_short": "codebert-base"},
                 {"fsuffix": "_fdeadbeef"}, {"resample_seed": 7}):
        lain = derive_ds_name(**{**FEATURISASI, **ubah})
        assert lain != nama, f"perubahan {ubah} tidak mengubah identitas artefak"


def test_t6_ketidakcocokan_featurisasi_terdeteksi():
    """T6 — checkpoint hanya sahih untuk dataset dengan featurisasi yang sama."""
    dataset = SimpleNamespace(fingerprint={"node_feat_dim": 843, "func_max_length": 1024,
                                           "num_classes": 26})
    assert check_compatible({"node_feat_dim": 843, "num_classes": 26}, dataset) == []

    masalah = check_compatible({"node_feat_dim": 768, "func_max_length": 5120}, dataset)
    assert len(masalah) == 2
    assert any("node_feat_dim" in m for m in masalah)


def test_t7_ruang_kelas_dibaca_dari_artefak_bukan_dari_pemanggil():
    """T7 — pembaruan model memakai vocab model dasar, sehingga id kelas lama tidak bergeser."""
    from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset

    vocab_dasar = {"not_vulnerable": 0, "CWE-125": 1, "CWE-787": 2, "CWE-401": 3}
    ds = SimpleNamespace(class_names=["not_vulnerable", "CWE-401", "CWE-125"],
                         _target_vocab=vocab_dasar)

    remap = CodeBERTGraphDataset._build_label_remap(ds)

    assert remap.tolist() == [0, 3, 1]
    assert ds.class_names == ["not_vulnerable", "CWE-125", "CWE-787", "CWE-401"]

    label_lama = torch.tensor([2])                      # CWE-125 pada urutan dataset baru
    assert remap[label_lama].item() == vocab_dasar["CWE-125"]


def test_t8_kelas_di_luar_vocab_sasaran_ditolak():
    """T8 — kelas yang tak dikenal model dasar gagal keras, bukan dipetakan diam-diam."""
    from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset

    ds = SimpleNamespace(class_names=["CWE-125", "CWE-999"],
                         _target_vocab={"not_vulnerable": 0, "CWE-125": 1})
    with pytest.raises(ValueError, match="target_vocab is missing"):
        CodeBERTGraphDataset._build_label_remap(ds)


# ── Tier lambat, menuntut Joern ───────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not Path(JOERN).is_dir(), reason="joern-cli tidak ditemukan")
def test_t9_pembangunan_data_dari_source_code(tmp_path):
    """T9 — kemampuan pembangunan data, source code menjadi dataset terbangun."""
    import json

    from gnn_vuln.core import build_dataset

    rows = [
        {"func": "int a(char *s){char b[8];strcpy(b,s);return 0;}",
         "CWE ID": "CWE-787", "target": 1, "flaw_lines": [1]},
        {"func": "int b(int *p,int n){int s=0;for(int i=0;i<n;i++)s+=p[i];return s;}",
         "CWE ID": "CWE-125", "target": 1, "flaw_lines": [1]},
        {"func": "int c(int x){return x+1;}",
         "CWE ID": "", "target": 0, "flaw_lines": []},
    ]
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    out = tmp_path / "processed" / "uji_modul.pt"
    out.parent.mkdir(parents=True)
    info = build_dataset(
        rows_path, out, featurization={"pretrained_lm": "microsoft/codebert-base",
                                       "add_func_tokens": False},
        joern_cli=JOERN, workers=1, storage="inmemory",
    )

    assert out.exists()
    assert info.n_graphs > 0
    assert info.class_names
