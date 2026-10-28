"""Lookup-table (MADDNESS-style) replacement for a linear layer, plus weight-quantization baselines.

A linear layer y = x W (x: [N, d], W: [d, m]) is replaced by:
  1. split x into C = d / L subvectors of length L;
  2. an encoder maps each subvector to one of K = 16 codes;
  3. y_hat[j] = sum_c T[c, code_c, j], where T is a table of shape [C, K, m].

Three encoders are implemented. All of them look only at the activation:
  - "nearest": exact nearest k-means prototype (16 * L multiplies per subvector; the accuracy ceiling);
  - "tree":    the MADDNESS 4-level decision tree (4 scalar comparisons per subvector);
  - "hash":    4 random hyperplanes per subspace, i.e. a 4-bit locality-sensitive hash.

After codes are fixed, the table is fit by ridge regression from one-hot codes to the exact layer
output on the calibration set, then quantized to 8 bits with one scale per output column and one
offset per (codebook, column), so the online sum is an integer add and the scale is applied once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

K = 16  # prototypes per subspace (4-bit codes)
TREE_DEPTH = 4


# ----------------------------------------------------------------------------- encoders


@dataclass
class Encoder:
    kind: str
    L: int
    C: int
    # nearest: prototypes [C, K, L]
    protos: torch.Tensor | None = None
    # tree: split_dims [C, 4] (one dim per level), thresholds [C, 4, 8] (per level, per bucket at that level)
    split_dims: torch.Tensor | None = None
    thresholds: torch.Tensor | None = None
    # hash: hyperplanes [C, 4, L], offsets [C, 4]
    planes: torch.Tensor | None = None
    plane_bias: torch.Tensor | None = None

    def to(self, device):
        for name in ("protos", "split_dims", "thresholds", "planes", "plane_bias"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, d] float -> codes [N, C] long."""
        N = x.shape[0]
        xs = x.reshape(N, self.C, self.L)
        if self.kind == "nearest":
            # argmin_k ||x - p_k||^2 = argmin_k (||p_k||^2 - 2 x.p_k); matmul form, chunked over rows,
            # so no [N, C, K, L] tensor is ever built.
            p2 = self.protos.pow(2).sum(-1)  # [C, K]
            out = torch.empty(N, self.C, dtype=torch.long, device=x.device)
            step = max(1, 2_000_000 // max(1, self.C * K))
            for s in range(0, N, step):
                xp = torch.einsum("ncl,ckl->nck", xs[s : s + step], self.protos)  # [n, C, K]
                out[s : s + step] = (p2.unsqueeze(0) - 2 * xp).argmin(-1)
            return out
        if self.kind == "tree":
            codes = torch.zeros(N, self.C, dtype=torch.long, device=x.device)
            cidx = torch.arange(self.C, device=x.device)
            for lvl in range(TREE_DEPTH):
                vals = xs[:, cidx, self.split_dims[:, lvl]]  # [N, C]
                thr = self.thresholds[cidx.unsqueeze(0), lvl, codes]  # [N, C]
                codes = codes * 2 + (vals > thr).long()
            return codes
        if self.kind == "hash":
            proj = torch.einsum("ncl,cbl->ncb", xs, self.planes) + self.plane_bias.unsqueeze(0)  # [N, C, 4]
            bits = (proj > 0).long()
            weights = torch.tensor([8, 4, 2, 1], device=x.device)
            return (bits * weights).sum(-1)
        raise ValueError(self.kind)


@torch.no_grad()
def _kmeans(xs: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    """xs: [C, N, L] -> prototypes [C, k, L], Lloyd's algorithm run independently per subspace."""
    C, N, L = xs.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:k].to(xs.device)
    protos = xs[:, idx, :].clone()  # [C, k, L]
    for _ in range(iters):
        p2 = protos.pow(2).sum(-1)  # [C, k]
        xp = torch.einsum("cnl,ckl->cnk", xs, protos)  # [C, N, k]
        assign = (p2.unsqueeze(1) - 2 * xp).argmin(-1)  # [C, N]
        del xp
        onehot = F.one_hot(assign, k).to(xs.dtype)  # [C, N, k]
        counts = onehot.sum(1)  # [C, k]
        sums = torch.einsum("cnk,cnl->ckl", onehot, xs)
        del onehot
        new = sums / counts.clamp(min=1).unsqueeze(-1)
        empty = counts == 0
        new[empty] = protos[empty]
        protos = new
    return protos


def _chunks(C: int, per_chunk_elems: int, N: int, L: int):
    """Yield (start, end) ranges over subspaces so each chunk holds about per_chunk_elems floats."""
    step = max(1, per_chunk_elems // max(1, N * L))
    for s in range(0, C, step):
        yield s, min(C, s + step)


@torch.no_grad()
def fit_nearest(x_cal: torch.Tensor, L: int) -> Encoder:
    N, d = x_cal.shape
    C = d // L
    xs = x_cal.reshape(N, C, L)
    protos = torch.cat([
        _kmeans(xs[:, a:b].transpose(0, 1).contiguous(), K) for a, b in _chunks(C, 20_000_000 * 16 // K, N, max(L, K))
    ])
    return Encoder("nearest", L, C, protos=protos)


@torch.no_grad()
def fit_hash(x_cal: torch.Tensor, L: int, seed: int = 0) -> Encoder:
    N, d = x_cal.shape
    C = d // L
    xs = x_cal.reshape(N, C, L)
    g = torch.Generator(device="cpu").manual_seed(seed)
    planes = torch.randn(C, TREE_DEPTH, L, generator=g).to(x_cal.device)
    planes = planes / planes.norm(dim=-1, keepdim=True)
    # bias so that each hyperplane passes through the subspace mean (centres the data)
    mean = xs.mean(0)  # [C, L]
    bias = -torch.einsum("cl,cbl->cb", mean, planes)
    return Encoder("hash", L, C, planes=planes, plane_bias=bias)


@torch.no_grad()
def fit_tree(x_cal: torch.Tensor, L: int) -> Encoder:
    """MADDNESS tree encoder, fit in chunks of subspaces to bound memory (see _fit_tree_block)."""
    N, d = x_cal.shape
    C = d // L
    xs = x_cal.reshape(N, C, L)
    dims, thrs = [], []
    for a, b in _chunks(C, 8_000_000, N, L):
        sd, th = _fit_tree_block(xs[:, a:b].transpose(0, 1).contiguous().float())
        dims.append(sd)
        thrs.append(th)
    return Encoder("tree", L, C, split_dims=torch.cat(dims), thresholds=torch.cat(thrs))


@torch.no_grad()
def _fit_tree_block(xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MADDNESS hash-function learning (Blalock & Guttag 2021, Alg. 1). xs: [C, N, L].

    At each of 4 levels, one split dimension is chosen per subspace (shared by all buckets at
    that level) and one threshold per bucket. The threshold for a bucket on dimension j is the
    one that minimizes the summed squared error of the two children (full subvector SSE), found
    from cumulative sums over rows sorted by that dimension. The dimension is the one with the
    lowest total loss over buckets.
    """
    C, N, L = xs.shape
    device = xs.device
    split_dims = torch.zeros(C, TREE_DEPTH, dtype=torch.long, device=device)
    thresholds = torch.zeros(C, TREE_DEPTH, 2 ** (TREE_DEPTH - 1), device=device)
    bucket = torch.zeros(C, N, dtype=torch.long, device=device)

    for lvl in range(TREE_DEPTH):
        nb = 2 ** lvl
        best_loss = torch.full((C,), float("inf"), device=device)
        best_dim = torch.zeros(C, dtype=torch.long, device=device)
        best_thr = torch.zeros(C, nb, device=device)
        for j in range(L):
            loss_j = torch.zeros(C, device=device)
            thr_j = torch.zeros(C, nb, device=device)
            for b in range(nb):
                mask = bucket == b  # [C, N]
                # Sort each subspace's rows in bucket b by dimension j. Work per subspace with a
                # masked large value so the sort pushes non-members to the end.
                key = xs[:, :, j].masked_fill(~mask, float("inf"))
                order = key.argsort(dim=1)  # [C, N]
                cnt = mask.sum(1)  # [C]
                xsorted = torch.gather(xs, 1, order.unsqueeze(-1).expand(-1, -1, L))  # [C, N, L]
                csum = xsorted.cumsum(1)
                csq = xsorted.pow(2).cumsum(1)
                n_left = torch.arange(1, N + 1, device=device).view(1, N, 1).float()
                tot_sum = torch.gather(csum, 1, (cnt - 1).clamp(min=0).view(C, 1, 1).expand(-1, 1, L))
                tot_sq = torch.gather(csq, 1, (cnt - 1).clamp(min=0).view(C, 1, 1).expand(-1, 1, L))
                n_right = cnt.view(C, 1, 1).float() - n_left
                sse_left = (csq - csum.pow(2) / n_left).sum(-1)  # [C, N]
                rsum = tot_sum - csum
                rsq = tot_sq - csq
                sse_right = (rsq - rsum.pow(2) / n_right.clamp(min=1)).sum(-1)
                sse = sse_left + sse_right
                # valid split positions: 1..cnt-1 members on the left
                pos = torch.arange(N, device=device).view(1, N)
                valid = (pos + 1 < cnt.view(C, 1)) & (pos >= 0)
                sse = sse.masked_fill(~valid, float("inf"))
                sse = torch.nan_to_num(sse, nan=float("inf"))
                best_pos = sse.argmin(1)  # [C]
                best_sse = sse.gather(1, best_pos.view(C, 1)).squeeze(1)
                # buckets with < 2 members cannot split: zero loss, threshold +inf (everything goes left)
                small = cnt < 2
                best_sse = best_sse.masked_fill(small, 0.0)
                sorted_key = torch.gather(key, 1, order)
                lo = sorted_key.gather(1, best_pos.view(C, 1)).squeeze(1)
                hi = sorted_key.gather(1, (best_pos + 1).clamp(max=N - 1).view(C, 1)).squeeze(1)
                thr = 0.5 * (lo + hi)
                thr = torch.where(small, torch.full_like(thr, float("inf")), thr)
                loss_j += best_sse
                thr_j[:, b] = thr
            better = loss_j < best_loss
            best_loss = torch.where(better, loss_j, best_loss)
            best_dim = torch.where(better, torch.full_like(best_dim, j), best_dim)
            best_thr = torch.where(better.unsqueeze(1), thr_j, best_thr)
        split_dims[:, lvl] = best_dim
        thresholds[:, lvl, :nb] = best_thr
        vals = xs[torch.arange(C, device=device).unsqueeze(1), torch.arange(N, device=device).unsqueeze(0), best_dim.unsqueeze(1)]
        thr_sel = best_thr.gather(1, bucket)  # [C, N]
        bucket = bucket * 2 + (vals > thr_sel).long()

    return split_dims, thresholds


def fit_encoder(kind: str, x_cal: torch.Tensor, L: int) -> Encoder:
    if kind == "nearest":
        return fit_nearest(x_cal, L)
    if kind == "tree":
        return fit_tree(x_cal, L)
    if kind == "hash":
        return fit_hash(x_cal, L)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- tables


@dataclass
class LUTLayer:
    enc: Encoder
    scale_in: torch.Tensor | None  # [d] SmoothQuant-style per-channel divisor, or None
    q: torch.Tensor  # [C, K, m] uint8 table entries
    col_scale: torch.Tensor  # [m]
    offset: torch.Tensor  # [C, m] per (codebook, column) offset
    bias: torch.Tensor | None

    def to(self, device):
        self.enc.to(device)
        for name in ("scale_in", "q", "col_scale", "offset", "bias"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        if self.scale_in is not None:
            x2 = x2 / self.scale_in
        codes = self.enc.encode(x2)  # [N, C]
        C, Kk, m = self.q.shape
        flat_idx = codes + torch.arange(C, device=x.device).unsqueeze(0) * Kk  # [N, C]
        table = self.q.reshape(C * Kk, m).float()
        # Gather-and-sum of 8-bit entries. Sums stay below 2^24 so float32 is exact (TF32 must be
        # off). For many rows a one-hot matmul is ~70x faster than embedding_bag on CUDA; for a few
        # rows (decoding) embedding_bag is fine.
        N = x2.shape[0]
        if N >= 32:
            acc = torch.empty(N, m, device=x.device)
            step = 4096
            for s in range(0, N, step):
                f = flat_idx[s : s + step]
                onehot = torch.zeros(f.shape[0], C * Kk, device=x.device).scatter_(1, f, 1.0)
                acc[s : s + step] = onehot @ table
        else:
            try:
                acc = F.embedding_bag(flat_idx, table, mode="sum")
            except Exception:
                acc = torch.zeros(N, m, device=x.device)
                for c in range(C):
                    acc += table[c * Kk + codes[:, c]]
        y = acc * self.col_scale + self.offset.sum(0)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], m).to(x.dtype)


@torch.no_grad()
def smooth_scales(x_cal: torch.Tensor, W: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant per-input-channel divisor s: x / s and W * s leave x W unchanged."""
    ax = x_cal.abs().amax(0).clamp(min=1e-5)  # [d]
    aw = W.abs().amax(1).clamp(min=1e-5)  # [d]
    return (ax.pow(alpha) / aw.pow(1 - alpha)).clamp(min=1e-5)


@torch.no_grad()
def bucket_means(codes: torch.Tensor, x: torch.Tensor, C: int, L: int) -> torch.Tensor:
    """Mean calibration subvector per (codebook, code): [C, K, L]. Works for any encoder."""
    N = x.shape[0]
    xs = x.reshape(N, C, L)
    sums = torch.zeros(C, K, L, device=x.device)
    counts = torch.zeros(C, K, device=x.device)
    ones = torch.ones(N, device=x.device)
    for c in range(C):
        sums[c].index_add_(0, codes[:, c], xs[:, c])
        counts[c].index_add_(0, codes[:, c], ones)
    return sums / counts.clamp(min=1).unsqueeze(-1)


@torch.no_grad()
def ridge_tables(codes: torch.Tensor, Y: torch.Tensor, C: int, lam: float = 1.0, T0: torch.Tensor | None = None) -> torch.Tensor:
    """Solve min_T || onehot(codes) T - Y ||^2 + lam' ||T - T0||^2 for T [C*K, m], with
    lam' = lam * N / K so that lam is relative to the typical number of rows per table entry
    (lam = 1 weights the prior like one extra bucket's worth of rows). T0 is the prior table
    (plain prototype dot products); None means shrink toward zero. Float32, on the device of
    `codes` (CPU on the laptop, GPU in the cloud)."""
    N = codes.shape[0]
    D = C * K
    dev = codes.device
    flat = codes + torch.arange(C, device=dev).unsqueeze(0) * K  # [N, C]
    Y = Y.to(dev).float()
    XtX = torch.zeros(D, D, device=dev)
    XtY = torch.zeros(D, Y.shape[1], device=dev)
    chunk = 2048
    for s in range(0, N, chunk):
        f = flat[s : s + chunk]
        X = torch.zeros(f.shape[0], D, device=dev)
        X.scatter_(1, f, 1.0)
        XtX.addmm_(X.T, X)
        XtY.addmm_(X.T, Y[s : s + chunk])
        del X
    reg = lam * N / K
    XtX.diagonal().add_(reg)
    if T0 is not None:
        XtY += reg * T0.to(dev).float()
    T = torch.linalg.solve(XtX, XtY)
    del XtX
    return T


@torch.no_grad()
def proto_tables(protos: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Plain MADDNESS tables: dot product of each prototype [C, K, L] with its slice of W [d, m]."""
    C, _, L = protos.shape
    Wsl = W.reshape(C, L, -1)  # [C, L, m]
    return torch.einsum("ckl,clm->ckm", protos, Wsl).reshape(C * K, -1)


@torch.no_grad()
def quantize_tables(T: torch.Tensor, C: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T [C*K, m] float -> (q [C, K, m] uint8, col_scale [m], offset [C, m]).

    y[j] = col_scale[j] * sum_c q[c, code_c, j] + sum_c offset[c, j].
    """
    T = T.reshape(C, K, -1)
    offset = T.amin(1)  # [C, m]
    shifted = T - offset.unsqueeze(1)  # >= 0
    col_scale = shifted.amax((0, 1)) / 255.0  # [m]
    col_scale = col_scale.clamp(min=1e-12)
    q = torch.round(shifted / col_scale).clamp(0, 255).to(torch.uint8)
    return q, col_scale, offset


@torch.no_grad()
def build_lut_layer(
    kind: str,
    L: int,
    x_cal: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor | None,
    smooth: bool,
    ridge: bool = True,
    lam: float = 1.0,
) -> LUTLayer:
    """x_cal: [N, d] calibration inputs (float32), W: [d, m] (float32), layer computes x W + bias."""
    x = x_cal.float()
    W = W.float()
    scale_in = None
    if smooth:
        scale_in = smooth_scales(x, W)
        x = x / scale_in
        W = W * scale_in.unsqueeze(1)
    enc = fit_encoder(kind, x, L)
    codes = enc.encode(x)
    # prior tables from the mean subvector of each bucket (= k-means prototypes for 'nearest')
    T0 = proto_tables(bucket_means(codes, x, enc.C, L), W)
    if ridge:
        T = ridge_tables(codes, x @ W, enc.C, lam, T0).to(x.device)
    else:
        T = T0
    q, col_scale, offset = quantize_tables(T, enc.C)
    return LUTLayer(enc, scale_in, q, col_scale, offset, bias)


# ----------------------------------------------------------------------------- weight-quant baselines


@torch.no_grad()
def rtn_quantize_weight(W: torch.Tensor, bits: int, group: int | None) -> torch.Tensor:
    """Round-to-nearest symmetric weight quantization. W: [d, m] (input dim first).

    bits=8, group=None: one scale per output column. bits=4, group=128: one scale per 128 input
    rows per output column (the usual INT4 recipe)."""
    W = W.float()
    qmax = 2 ** (bits - 1) - 1
    if group is None:
        scale = W.abs().amax(0).clamp(min=1e-8) / qmax  # [m]
        return torch.round(W / scale).clamp(-qmax - 1, qmax) * scale
    d, m = W.shape
    Wg = W.reshape(d // group, group, m)
    scale = Wg.abs().amax(1, keepdim=True).clamp(min=1e-8) / qmax
    return (torch.round(Wg / scale).clamp(-qmax - 1, qmax) * scale).reshape(d, m)


def bytes_per_token(d: int, m: int, method: str, L: int | None = None) -> float:
    """Bytes read per token at batch size 1 for one projection."""
    if method == "fp16":
        return 2.0 * d * m
    if method == "int8":
        return 1.0 * d * m
    if method == "int4":
        return 0.5 * d * m
    if method == "lut":
        return d * m / L  # one row of m bytes per codebook, C = d / L codebooks
    raise ValueError(method)


def storage_bytes(d: int, m: int, method: str, L: int | None = None) -> float:
    if method == "lut":
        return K * d * m / L
    return bytes_per_token(d, m, method)
