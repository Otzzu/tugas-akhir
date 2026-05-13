"""
Correctness + speed check: StmtHead._score_loop vs _score_vectorized.
Run: uv run python scripts/test_stmthead_vectorized.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # must be before CUDA init

import time
import torch
from gnn_vuln.models.heads import StmtHead

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

WARMUP = 5
REPS   = 50

def bench(head, impl: bool, h, batch, node_line, lm_hidden, func_token_lines):
    head._vectorized = impl
    with torch.no_grad():
        for _ in range(WARMUP):
            head.score(h, batch, node_line, lm_hidden, func_token_lines)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            head.score(h, batch, node_line, lm_hidden, func_token_lines)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / REPS * 1000  # ms per call

def run_test(mode: str, use_lm: bool, B=8, N=200, D=256, LM_D=768, L=512):
    print(f"\n[{mode}] use_lm={use_lm}")
    head = StmtHead(hidden_dim=D, lm_dim=LM_D if use_lm else 0,
                    localization_encoder=mode).to(device)
    head.eval()

    h         = torch.randn(N, D, device=device)
    batch     = torch.randint(0, B, (N,), device=device).sort().values
    node_line = torch.randint(-1, 50, (N,), device=device)

    lm_hidden = func_token_lines = None
    if use_lm:
        lm_hidden        = torch.randn(B, L, LM_D, device=device)
        func_token_lines = torch.randint(-1, 50, (B, L), device=device)

    with torch.no_grad():
        head._vectorized = False
        out_loop = head.score(h, batch, node_line, lm_hidden, func_token_lines)
        head._vectorized = True
        out_vec  = head.score(h, batch, node_line, lm_hidden, func_token_lines)

    assert len(out_loop) == len(out_vec) == B, "length mismatch"
    max_err = 0.0
    for i, (a, b_) in enumerate(zip(out_loop, out_vec)):
        if a.numel() == 0 and b_.numel() == 0:
            continue
        assert a.shape == b_.shape, f"graph {i} shape mismatch: {a.shape} vs {b_.shape}"
        err = (a - b_).abs().max().item()
        max_err = max(max_err, err)

    t_loop = bench(head, False, h, batch, node_line, lm_hidden, func_token_lines)
    t_vec  = bench(head, True,  h, batch, node_line, lm_hidden, func_token_lines)
    ok     = max_err < 1e-4
    speedup = t_loop / t_vec if t_vec > 0 else float('inf')
    print(f"  max abs error : {max_err:.2e}  {'PASS' if ok else 'FAIL'}")
    print(f"  loop          : {t_loop:.3f} ms/call")
    print(f"  vectorized    : {t_vec:.3f} ms/call")
    print(f"  speedup       : {speedup:.2f}x")
    return ok

results = []
results.append(run_test("gnn",  use_lm=False))
results.append(run_test("gnn",  use_lm=True))
results.append(run_test("lm",   use_lm=True))
results.append(run_test("both", use_lm=True))

print("\n[edge] all nodes invalid (line=-1)")
head = StmtHead(hidden_dim=256, lm_dim=0, localization_encoder="gnn").to(device)
h = torch.randn(20, 256, device=device)
batch = torch.zeros(20, dtype=torch.long, device=device)
node_line = torch.full((20,), -1, dtype=torch.long, device=device)
with torch.no_grad():
    head._vectorized = False; ol = head.score(h, batch, node_line)
    head._vectorized = True;  ov = head.score(h, batch, node_line)
assert ol[0].numel() == 0 and ov[0].numel() == 0
print("  PASS (empty output)")
results.append(True)

print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'}")

# ── Deterministic mode check ───────────────────────────────────────────────────
# scatter_add_() and scatter_reduce_(amax) are both deterministic in PyTorch 2.x
# under use_deterministic_algorithms(True). This verifies no RuntimeError is raised
# and that repeated calls produce bit-exact output.
print("\n[deterministic] strict mode — loop + vectorized (correctness + timing vs non-det)")
det_results = []
for mode, use_lm in [("gnn", False), ("gnn", True), ("lm", True), ("both", True)]:
    B, N, D, LM_D, L = 4, 100, 256, 768, 256
    head = StmtHead(hidden_dim=D, lm_dim=LM_D if use_lm else 0,
                    localization_encoder=mode).to(device).eval()
    h         = torch.randn(N, D, device=device)
    batch     = torch.randint(0, B, (N,), device=device).sort().values
    node_line = torch.randint(-1, 50, (N,), device=device)
    lm_hidden = func_token_lines = None
    if use_lm:
        lm_hidden        = torch.randn(B, L, LM_D, device=device)
        func_token_lines = torch.randint(-1, 50, (B, L), device=device)
    print(f"\n  [{mode}] use_lm={use_lm}")
    try:
        for impl, label in [(False, "loop"), (True, "vec ")]:
            # correctness: bit-exact repeat under deterministic
            torch.use_deterministic_algorithms(True, warn_only=False)
            with torch.no_grad():
                head._vectorized = impl
                o1 = head.score(h, batch, node_line, lm_hidden, func_token_lines)
                o2 = head.score(h, batch, node_line, lm_hidden, func_token_lines)
            max_err = max(
                (a - b).abs().max().item()
                for a, b in zip(o1, o2) if a.numel() > 0
            ) if any(a.numel() > 0 for a in o1) else 0.0
            ok = max_err == 0.0
            det_results.append(ok)

            # timing: det vs non-det
            t_det  = bench(head, impl, h, batch, node_line, lm_hidden, func_token_lines)
            torch.use_deterministic_algorithms(False)
            t_ndet = bench(head, impl, h, batch, node_line, lm_hidden, func_token_lines)
            overhead = (t_det / t_ndet - 1) * 100 if t_ndet > 0 else float('nan')
            print(f"    {label}: bit-exact={'PASS' if ok else 'FAIL'}  "
                  f"det={t_det:.3f}ms  non-det={t_ndet:.3f}ms  overhead={overhead:+.1f}%")
    except RuntimeError as e:
        torch.use_deterministic_algorithms(False)
        print(f"    FAIL RuntimeError: {e}")
        det_results.append(False)

print(f"\n[deterministic] {'ALL PASS' if all(det_results) else 'SOME FAILED'}")
