"""Experiment 2: acceptance under naive row choice.

Replace every linear projection inside the drafter's transformer blocks (q, k, v, o, gate, up, down;
embeddings and the output head stay exact) with lookup tables, and measure how often the target
accepts the drafter's tokens. Baselines: the exact drafter and the same drafter with 4-bit weights.

Two acceptance measurements:
  * teacher-forced, on natural text: alpha_sample = E[sum_v min(p_target(v), q_draft(v))], the
    per-token acceptance probability of speculative sampling (Leviathan et al. 2023), and
    alpha_greedy = P(argmax q == argmax p). Also KL(p_target || q_draft) and the drafter's NLL.
  * draft rounds: the drafter generates gamma tokens greedily from a natural-text prefix, the target
    scores them in one pass, and we count the leading run of agreements. Reported as accepted tokens
    per round and the geometric estimate alpha_hat = accepted / (accepted + rounds with a rejection).
    This includes the compounding of the drafter's errors along its own continuation.

Usage (cloud GPU):
  python experiments/exp2_acceptance.py --target meta-llama/Llama-3.2-3B --out results/exp2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import lut  # noqa: E402
from exp1_single_layer import Replaced, load_tokens  # noqa: E402

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
# projections that read the same input tensor as another one (calibration rows are shared)
SHARED = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj",
          "mlp.up_proj": "mlp.gate_proj"}
UNIQUE = [p for p in PROJ if p not in SHARED]
SETS = {
    "all": PROJ,
    "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "down": ["mlp.down_proj"],
}
# A set may be restricted to a layer range with "@a-b" (inclusive), e.g. "down@0-3". The default
# list is a compounding sweep: how fast does acceptance fall as more projections are replaced?
DEFAULT_CONFIGS = [
    "exact",
    "int4:all",
    "lut:nearest:8:down@8-8",
    "lut:nearest:8:down@0-3",
    "lut:nearest:8:down@0-7",
    "lut:nearest:8:down",
    "lut:nearest:8:mlp",
    "lut:nearest:8:attn",
    "lut:nearest:8:all",
    "lut:tree:8:down",
    "lut:hash:8:down",
    "lut:nearest:16:down",
    "lut:nearest:32:down",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_mod(model, layer: int, name: str) -> nn.Module:
    mod = model.model.layers[layer]
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def set_mod(model, layer: int, name: str, new: nn.Module):
    parent = model.model.layers[layer]
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


# ----------------------------------------------------------------------------- calibration


@torch.no_grad()
def collect_all(model, tokens, batch, device, rows_per_input, seed, layer_ids=None):
    """Inputs to q_proj, o_proj, gate_proj and down_proj of every layer (the four distinct input
    tensors per block), subsampled to rows_per_input rows each, stored float16 on CPU.
    With layer_ids, only those layers are recorded (used for sequential calibration, where the
    model already contains the replacements for earlier layers)."""
    n_layers = len(model.model.layers)
    layer_ids = list(range(n_layers)) if layer_ids is None else layer_ids
    store = {(l, n): [] for l in layer_ids for n in UNIQUE}
    captured = {}
    hooks = []
    for l in layer_ids:
        for n in UNIQUE:
            def hook(_m, inp, _out, key=(l, n)):
                captured[key] = inp[0].detach().reshape(-1, inp[0].shape[-1])
            hooks.append(get_mod(model, l, n).register_forward_hook(hook))
    g = torch.Generator().manual_seed(seed)
    n_batches = math.ceil(tokens.shape[0] / batch)
    per_batch = math.ceil(rows_per_input / n_batches)
    for s in range(0, tokens.shape[0], batch):
        model(tokens[s : s + batch].to(device))
        n_rows = next(iter(captured.values())).shape[0]
        idx = torch.randperm(n_rows, generator=g)[:per_batch].to(device)
        for key, x in captured.items():
            store[key].append(x[idx].to(torch.float16).cpu())
        captured.clear()
    for h in hooks:
        h.remove()
    return {k: torch.cat(v)[:rows_per_input] for k, v in store.items()}


# ----------------------------------------------------------------------------- replacements


class Swapper:
    """Holds the drafter's original modules and swaps replacements in and out."""

    def __init__(self, model):
        self.model = model
        self.n_layers = len(model.model.layers)
        self.originals = {(l, n): get_mod(model, l, n) for l in range(self.n_layers) for n in PROJ}
        self.active: list[tuple[int, str]] = []
        self.saved_weights: dict[tuple[int, str], torch.Tensor] = {}

    def apply_lut(self, layers: dict, names: list[str], layer_ids: list[int], device):
        for l in layer_ids:
            for n in names:
                layer = layers[(l, n)].to(device)
                set_mod(self.model, l, n, Replaced(layer.forward))
                self.active.append((l, n))

    def apply_int4(self, names: list[str], layer_ids: list[int]):
        for l in layer_ids:
            for n in names:
                lin = self.originals[(l, n)]
                self.saved_weights[(l, n)] = lin.weight.detach().clone().cpu()
                W = lin.weight.detach().float().T.contiguous()  # [d, m]
                Wq = lut.rtn_quantize_weight(W, 4, min(128, W.shape[0]))
                lin.weight.data.copy_(Wq.T.to(lin.weight.dtype))

    def restore(self, layers: dict | None = None):
        for (l, n) in self.active:
            set_mod(self.model, l, n, self.originals[(l, n)])
            if layers is not None:
                layers[(l, n)].to("cpu")
        self.active = []
        for (l, n), w in self.saved_weights.items():
            self.originals[(l, n)].weight.data.copy_(w.to(self.originals[(l, n)].weight.device))
        self.saved_weights = {}


@torch.no_grad()
def fit_layers(swapper: Swapper, cal, kind: str, L: int, names: list[str], layer_ids: list[int], lam: float,
               device, cache: dict):
    """Fit lookup tables for the given projections of the given layers; results go to `cache` (CPU)."""
    t0 = time.time()
    for l in layer_ids:
        for n in names:
            if (l, n) in cache:
                continue
            x = cal[(l, SHARED.get(n, n))].to(device).float()
            lin = swapper.originals[(l, n)]
            W = lin.weight.detach().float().T.contiguous()
            b = lin.bias.detach().float() if lin.bias is not None else None
            layer = lut.build_lut_layer(kind, L, x, W, b, smooth=False, ridge=True, lam=lam)
            cache[(l, n)] = layer.to("cpu")
            del x, W
        log(f"  fit {kind} L={L}: layer {l} done ({time.time() - t0:.0f}s)")
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- evaluation


@torch.no_grad()
def teacher_forced(target, drafter, tokens, batch, device):
    tot = dict(alpha=0.0, greedy=0.0, kl=0.0, nll=0.0, n=0)
    for s in range(0, tokens.shape[0], batch):
        ids = tokens[s : s + batch].to(device)
        tgt = ids[:, 1:]
        lp_t = F.log_softmax(target(ids).logits[:, :-1].float(), -1)
        lp_d = F.log_softmax(drafter(ids).logits[:, :-1].float(), -1)
        p_t = lp_t.exp()
        tot["alpha"] += torch.minimum(p_t, lp_d.exp()).sum(-1).sum().item()
        tot["greedy"] += (lp_t.argmax(-1) == lp_d.argmax(-1)).sum().item()
        tot["kl"] += (p_t * (lp_t - lp_d)).sum(-1).sum().item()
        tot["nll"] += F.nll_loss(lp_d.reshape(-1, lp_d.shape[-1]), tgt.reshape(-1), reduction="sum").item()
        tot["n"] += tgt.numel()
        del lp_t, lp_d, p_t
    n = tot["n"]
    return dict(alpha_sample=tot["alpha"] / n, alpha_greedy=tot["greedy"] / n, kl=tot["kl"] / n, nll=tot["nll"] / n)


@torch.no_grad()
def draft_rounds(target, drafter, tokens, gamma, n_rounds, batch, device, seed, pad_id):
    g = torch.Generator().manual_seed(seed)
    accepted, rounds, rejected, hist = 0, 0, 0, [0] * (gamma + 1)
    t0 = time.time()
    for r in range(0, n_rounds, batch):
        b = min(batch, n_rounds - r)
        seq_idx = torch.randint(0, tokens.shape[0], (b,), generator=g)
        plen = int(torch.randint(32, tokens.shape[1] - gamma, (1,), generator=g))
        prefix = tokens[seq_idx, :plen].to(device)
        out = drafter.generate(prefix, max_new_tokens=gamma, min_new_tokens=gamma, do_sample=False,
                               pad_token_id=pad_id, attention_mask=torch.ones_like(prefix))
        draft = out[:, plen : plen + gamma]
        t_tok = target(out[:, : plen + gamma]).logits[:, plen - 1 : plen + gamma - 1].argmax(-1)
        acc = (t_tok == draft).long().cumprod(-1).sum(-1)  # leading run of agreements
        for a in acc.tolist():
            hist[a] += 1
        accepted += acc.sum().item()
        rejected += (acc < gamma).sum().item()
        rounds += b
    alpha_hat = accepted / max(1, accepted + rejected)
    return dict(accepted_per_round=accepted / rounds, alpha_hat=alpha_hat, rounds=rounds,
                hist=hist, round_seconds=time.time() - t0)


def drafter_bytes(swapper: Swapper, method: str, L: int | None, names: list[str], layer_ids: list[int],
                  embed_dim: int, vocab: int):
    """Bytes read per token (batch 1) and stored for the drafter's block projections, plus the
    unreplaced ones in 16 bits. The output head (vocab x d, 16-bit) is listed separately."""
    read = store = 0.0
    for (l, n), lin in swapper.originals.items():
        m, d = lin.weight.shape
        if n in names and l in layer_ids and method != "fp16":
            read += lut.bytes_per_token(d, m, method, L)
            store += lut.storage_bytes(d, m, method, L)
        else:
            read += 2.0 * d * m
            store += 2.0 * d * m
    head = 2.0 * embed_dim * vocab
    return read, store, head


# ----------------------------------------------------------------------------- main


def parse_set(spec: str, n_layers: int):
    """'down' -> all layers; 'down@0-3' -> layers 0..3 inclusive."""
    name, _, rng = spec.partition("@")
    if rng:
        a, b = (int(v) for v in rng.split("-"))
        layer_ids = list(range(a, b + 1))
    else:
        layer_ids = list(range(n_layers))
    return SETS[name], layer_ids


def parse_config(s: str, n_layers: int):
    parts = s.split(":")
    if parts[0] == "exact":
        return dict(method="fp16", encoder="", L=None, names=[], layer_ids=[], set="")
    if parts[0] == "int4":
        names, layer_ids = parse_set(parts[1], n_layers)
        return dict(method="int4", encoder="", L=None, names=names, layer_ids=layer_ids, set=parts[1])
    if parts[0] == "lut":
        names, layer_ids = parse_set(parts[3], n_layers)
        return dict(method="lut", encoder=parts[1], L=int(parts[2]), names=names, layer_ids=layer_ids, set=parts[3])
    raise ValueError(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--target", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--cal-tokens", type=int, default=200_000)
    ap.add_argument("--cal-rows", type=int, default=16_384)
    ap.add_argument("--eval-tokens", type=int, default=16_384)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=512)
    ap.add_argument("--round-batch", type=int, default=16)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--sequential", action="store_true",
                    help="calibrate layer by layer on the inputs produced with earlier layers already replaced")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/exp2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False  # table sums are exact integers in float32 only without TF32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.drafter)
    drafter = AutoModelForCausalLM.from_pretrained(args.drafter, dtype=dtype).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    assert drafter.config.vocab_size == target.config.vocab_size, "drafter and target must share a tokenizer"
    pad_id = tok.eos_token_id

    cal = load_tokens(tok, "train", args.cal_tokens, args.seq_len, args.seed)
    ev = load_tokens(tok, "test", args.eval_tokens, args.seq_len, args.seed + 1)
    log(f"calibration {cal.numel()} tokens, eval {ev.numel()} tokens")

    swapper = Swapper(drafter)
    t0 = time.time()
    cal_rows = collect_all(drafter, cal, args.batch, device, args.cal_rows, args.seed)
    log(f"collected {args.cal_rows} rows for {len(cal_rows)} inputs ({time.time() - t0:.0f}s, "
        f"{sum(v.numel() * 2 for v in cal_rows.values()) / 1e9:.1f} GB on CPU)")
    if not args.sequential:
        del cal

    fields = ["config", "method", "encoder", "L", "set", "n_replaced", "alpha_sample", "alpha_greedy", "kl", "nll",
              "accepted_per_round", "alpha_hat", "rounds", "hist", "read_mb", "store_mb", "head_mb", "fit_seconds",
              "eval_seconds"]
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    meta = dict(vars(args), device=str(device), configs_done=[])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    cache_key, cache = None, {}
    for cfg_s in args.configs:
        cfg = parse_config(cfg_s, swapper.n_layers)
        if args.sequential and cfg["method"] == "lut":
            cfg_s = cfg_s + "+seq"
        log(f"== {cfg_s}")
        t_fit = time.time()
        if cfg["method"] == "lut" and args.sequential:
            # layer by layer: record this layer's inputs with earlier layers already replaced, fit, swap in
            cache_key, cache = None, {}
            for l in cfg["layer_ids"]:
                rows = collect_all(drafter, cal, args.batch, device, args.cal_rows, args.seed, layer_ids=[l])
                fit_layers(swapper, rows, cfg["encoder"], cfg["L"], cfg["names"], [l], args.lam, device, cache)
                swapper.apply_lut(cache, cfg["names"], [l], device)
                del rows
        elif cfg["method"] == "lut":
            key = (cfg["encoder"], cfg["L"])
            if key != cache_key:
                cache_key, cache = key, {}
            fit_layers(swapper, cal_rows, cfg["encoder"], cfg["L"], cfg["names"], cfg["layer_ids"], args.lam, device, cache)
            swapper.apply_lut(cache, cfg["names"], cfg["layer_ids"], device)
        elif cfg["method"] == "int4":
            swapper.apply_int4(cfg["names"], cfg["layer_ids"])
        fit_seconds = time.time() - t_fit

        t_ev = time.time()
        tf = teacher_forced(target, drafter, ev, args.batch, device)
        dr = draft_rounds(target, drafter, ev, args.gamma, args.rounds, args.round_batch, device, args.seed, pad_id)
        eval_seconds = time.time() - t_ev
        swapper.restore(cache if cfg["method"] == "lut" else None)
        torch.cuda.empty_cache()

        read, store, head = drafter_bytes(swapper, cfg["method"], cfg["L"], cfg["names"], cfg["layer_ids"],
                                          drafter.config.hidden_size, drafter.config.vocab_size)
        row = dict(config=cfg_s, method=cfg["method"], encoder=cfg["encoder"], L=cfg["L"] or "", set=cfg["set"],
                   n_replaced=len(cfg["names"]) * len(cfg["layer_ids"]), **tf,
                   accepted_per_round=dr["accepted_per_round"], alpha_hat=dr["alpha_hat"], rounds=dr["rounds"],
                   hist=" ".join(map(str, dr["hist"])), read_mb=read / 1e6, store_mb=store / 1e6, head_mb=head / 1e6,
                   fit_seconds=fit_seconds, eval_seconds=eval_seconds)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        log(f"{cfg_s:22s} alpha_sample={tf['alpha_sample']:.3f} alpha_greedy={tf['alpha_greedy']:.3f} "
            f"KL={tf['kl']:.3f} nll={tf['nll']:.3f} | gamma={args.gamma}: {dr['accepted_per_round']:.2f} tok/round "
            f"alpha_hat={dr['alpha_hat']:.3f} hist={dr['hist']} | read {read/1e6:.0f} MB (+{head/1e6:.0f} head) "
            f"fit {fit_seconds:.0f}s eval {eval_seconds:.0f}s")
        meta["configs_done"].append(cfg_s)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log("done")


if __name__ == "__main__":
    main()
