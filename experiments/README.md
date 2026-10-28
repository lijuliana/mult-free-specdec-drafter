# Experiments

Setup:

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
hf auth login   # token with access to meta-llama/Llama-3.2-1B and Llama-3.1-8B
```

## Experiment 1: single-layer reconstruction

`exp1_single_layer.py` replaces one linear projection of the drafter and measures the damage,
first at the layer and then at the model's output.

**Design**

- Model: Llama-3.2-1B (base). Layer 8 of 16. Two projections: `mlp.down_proj` (8192 → 2048, the
  widest input and the one most exposed to activation outliers) and `self_attn.o_proj` (2048 → 2048).
- Calibration: 1M tokens of WikiText-103 train, from which 100K input rows of the projection are
  sampled uniformly. Evaluation: 32K tokens of WikiText-103 test, disjoint from calibration.
- Methods, every one applied to the same projection with the rest of the model exact:
  - `int8`: round-to-nearest weights, one scale per output column.
  - `int4`: round-to-nearest weights, one scale per 128 input rows per output column.
  - `lut`: lookup tables at subvector length L in {8, 16, 32}, 16 prototypes per subspace, 8-bit
    entries. Tables start as the dot product of each bucket's mean subvector with its slice of W and
    are then refit by ridge regression against the exact layer output on the calibration rows, with
    the penalty shrinking toward those prototype tables (`--lam`, default 3, relative to the number
    of rows per table entry; a plain ridge toward zero overfits once C·16 approaches the row count
    and made tables worse than the prototype dot products). Crossed with three encoders (how a
    subvector is assigned to a prototype) and with
    SmoothQuant-style per-channel scaling on or off (folded into the weights, free at run time):
    - `nearest`: exact nearest k-means prototype. The accuracy ceiling for a fixed codebook.
    - `tree`: the MADDNESS 4-level decision tree, learned as in Blalock and Guttag (2021).
    - `hash`: 4 random hyperplanes per subspace through the subspace mean, a 4-bit
      locality-sensitive hash. The naive baseline from the lecture framing.
- Metrics:
  - `rel_err`: ||y_hat − y||_F / ||y||_F on 20K held-out input rows of the projection.
  - `kl`: mean over evaluation positions of KL(p_exact || p_replaced) between the full model's
    next-token distributions with and without the replacement.
  - `top1`: fraction of positions where the replaced model's argmax equals the exact model's. This is
    the greedy acceptance rate a drafter would get if the target were the exact model.
  - `nll`: the replaced model's own next-token loss on the evaluation tokens.
  - `bytes_per_token`, `storage_bytes`: for the projection, at batch size 1.
- Gate from the proposal: at L = 8 the KL cost of the best lookup configuration is within 2× that
  of `int4`. If it is not, outlier handling (rotations, per-channel grouping) comes before anything else.

**What the comparisons answer**

- `nearest` vs `tree` vs `hash` at the same L: how much accuracy the cheap encoders give up. This
  is the baseline gap that target-guided row choice (experiment 3) has to close.
- `smooth` on vs off: whether activation outliers are the dominant error source.
- L = 8 vs 16 vs 32: the accuracy cost of reading 2× and 4× fewer bytes.
- `lut` vs `int4` at matched bytes: tables at L = 32 store as many bytes as INT4 and read 1/4 as
  many per token; tables at L = 8 read 1/16 of FP16 but store as much as FP16.

**Run**

```
.venv/bin/python experiments/exp1_single_layer.py            # full design, Llama-3.2-1B
.venv/bin/python experiments/exp1_single_layer.py --model Qwen/Qwen3-0.6B --layer 14   # ungated stand-in
```

Results go to `results/exp1/results.csv` (one row per configuration) and `results/exp1/meta.json`.

Defaults are CPU-only with 4 threads and an in-process abort at 6 GB resident memory
(`--max-rss-gb`), because GPU runs on a laptop stalled the machine twice. Pass `--device cuda
--dtype bfloat16 --threads 8 --max-rss-gb 40` on a cloud GPU. The ridge solve always runs on CPU.

`diag_layer.py` is the layer-level diagnostic (no model forward passes after collecting
activations): it isolates the error contributed by table rounding, the ridge fit and its penalty,
subvector length and prototype count.

### First numbers (stand-in model Qwen3-0.6B, layer 14, 10K calibration rows)

Relative output error of the MLP down-projection on held-out rows. INT4: 0.14, INT3: 0.33, INT2: 0.81.
Tables at L = 8 with the nearest-prototype encoder: 0.58 (tree 0.67, hash 0.71); L = 4: 0.48;
L = 16: 0.67; 256 prototypes at L = 8: 0.44. 8-bit table rounding adds nothing measurable; outlier
scaling does not help, and the top 8 channels hold only 8% of the max-energy. The error is the
resolution of 4-bit codes over 8 dimensions, not an outlier or implementation artifact.
