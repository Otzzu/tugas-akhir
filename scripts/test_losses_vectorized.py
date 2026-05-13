"""
Correctness + speed check: mil_loss and ranking_loss before/after vectorization.
Run: uv run python scripts/test_losses_vectorized.py
"""
import time
import torch
import torch.nn.functional as F
from gnn_vuln.training.losses import mil_loss, ranking_loss

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

WARMUP = 10
REPS   = 100
B      = 32   # batch size
K      = 3    # mil_k
MAX_STMTS = 60  # avg stmts per graph


def make_inputs(B=B, flaw_ratio=0.5, seed=0):
    torch.manual_seed(seed)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    n_vuln = int(B * flaw_ratio)
    labels[:n_vuln] = torch.randint(1, 10, (n_vuln,), device=device)

    # Build node-level tensors first, then derive stmt_scores sizes from unique lines.
    # StmtHead returns one score per unique (graph, line) — sizes must match.
    nodes_per_graph = torch.randint(20, MAX_STMTS * 3, (B,)).tolist()
    batch_idx_list, node_line_list, flaw_list = [], [], []
    stmt_scores = []

    for b, n_nodes in enumerate(nodes_per_graph):
        n_lines = torch.randint(5, MAX_STMTS, (1,)).item()
        lines = torch.randint(0, n_lines, (n_nodes,), device=device)
        unique_lines = lines.unique()
        n_unique = unique_lines.shape[0]

        batch_idx_list.append(torch.full((n_nodes,), b, dtype=torch.long, device=device))
        node_line_list.append(lines)

        flaw = torch.zeros(n_nodes, dtype=torch.float, device=device)
        if b < n_vuln and n_nodes > 0:
            flaw[:max(1, n_nodes // 4)] = 1.0
        flaw_list.append(flaw)

        # one score per unique stmt — matches StmtHead output shape
        stmt_scores.append(torch.randn(n_unique, device=device))

    batch_idx = torch.cat(batch_idx_list)
    node_line  = torch.cat(node_line_list)
    flaw_mask  = torch.cat(flaw_list)

    return stmt_scores, labels, batch_idx, node_line, flaw_mask


def bench_fn(fn, *args, **kwargs):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            fn(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1000


# ── Reference implementations (original loop versions) ───────────────────────

def mil_loss_loop(stmt_scores_list, labels, k):
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0
    for scores, label in zip(stmt_scores_list, labels):
        if len(scores) == 0:
            continue
        actual_k = min(k, len(scores))
        _, topk_idx = scores.topk(actual_k)
        topk_scores = scores[topk_idx]
        binary_label = float(label.item() > 0)
        pseudo = torch.full((actual_k,), binary_label, device=device)
        total = total + F.binary_cross_entropy_with_logits(topk_scores, pseudo)
        count += 1
    return total / count if count > 0 else total


def ranking_loss_loop(stmt_scores_list, batch_idx, node_line, flaw_line_mask, labels, margin=1.0):
    device = labels.device
    total = torch.tensor(0.0, device=device)
    count = 0
    for b, (scores, label) in enumerate(zip(stmt_scores_list, labels)):
        if label.item() == 0 or len(scores) == 0:
            continue
        mask = batch_idx == b
        lines_b = node_line[mask]
        flaw_b  = flaw_line_mask[mask]
        valid = lines_b >= 0
        if not valid.any():
            continue
        lines_b = lines_b[valid]
        flaw_b  = flaw_b[valid]
        unique_lines = lines_b.unique(sorted=True)
        flaw_flags = torch.stack([
            flaw_b[lines_b == line].max() for line in unique_lines
        ]).bool()
        if not flaw_flags.any() or flaw_flags.all():
            continue
        flaw_scores = scores[flaw_flags]
        safe_scores = scores[~flaw_flags]
        diff = flaw_scores.unsqueeze(1) - safe_scores.unsqueeze(0)
        total = total + F.relu(margin - diff).mean()
        count += 1
    return total / count if count > 0 else total


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_mil():
    print("\n── mil_loss ──")
    stmt_scores, labels, _, _, _ = make_inputs()

    with torch.no_grad():
        out_loop = mil_loss_loop(stmt_scores, labels, K)
        out_vec  = mil_loss(stmt_scores, labels, K)

    err = (out_loop - out_vec).abs().item()
    ok  = err < 1e-4
    print(f"  loop result : {out_loop.item():.6f}")
    print(f"  vec  result : {out_vec.item():.6f}")
    print(f"  abs error   : {err:.2e}  {'PASS' if ok else 'FAIL'}")

    t_loop = bench_fn(mil_loss_loop, stmt_scores, labels, K)
    t_vec  = bench_fn(mil_loss,      stmt_scores, labels, K)
    speedup = t_loop / t_vec if t_vec > 0 else float('inf')
    print(f"  loop        : {t_loop:.3f} ms/call")
    print(f"  vectorized  : {t_vec:.3f} ms/call")
    print(f"  speedup     : {speedup:.2f}x")
    return ok


def test_ranking():
    print("\n── ranking_loss ──")
    stmt_scores, labels, batch_idx, node_line, flaw_mask = make_inputs()

    with torch.no_grad():
        out_loop = ranking_loss_loop(stmt_scores, batch_idx, node_line, flaw_mask, labels)
        out_vec  = ranking_loss(stmt_scores, batch_idx, node_line, flaw_mask, labels)

    err = (out_loop - out_vec).abs().item()
    ok  = err < 1e-4
    print(f"  loop result : {out_loop.item():.6f}")
    print(f"  vec  result : {out_vec.item():.6f}")
    print(f"  abs error   : {err:.2e}  {'PASS' if ok else 'FAIL'}")

    t_loop = bench_fn(ranking_loss_loop, stmt_scores, batch_idx, node_line, flaw_mask, labels)
    t_vec  = bench_fn(ranking_loss,      stmt_scores, batch_idx, node_line, flaw_mask, labels)
    speedup = t_loop / t_vec if t_vec > 0 else float('inf')
    print(f"  loop        : {t_loop:.3f} ms/call")
    print(f"  vectorized  : {t_vec:.3f} ms/call")
    print(f"  speedup     : {speedup:.2f}x")
    return ok


results = [test_mil(), test_ranking()]
print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'}")
