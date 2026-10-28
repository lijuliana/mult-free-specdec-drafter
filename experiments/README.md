# Experiments

Code and results for the starter experiments of [PROPOSAL.md](../PROPOSAL.md). Raw outputs are in
[`results/`](../results).

| Experiment | Status | Headline |
| --- | --- | --- |
| 1. Single-layer reconstruction | Done (Llama-3.2-1B) | Best table at L = 8 costs about 20× the KL of 4-bit weights. Gate failed. |
| 2. Acceptance under naive row choice | Done (1B drafting for 3B) | One table layer costs 2 points of acceptance; tables in every layer collapse the drafter. Gate failed. |
| 4. Fused kernel | First pass done (A10G) | Lookup tables run all 112 projections 5.8× faster than BF16. |
| 3, 5, 6, 7 | Not started | |

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
hf auth login   # token with access to meta-llama/Llama-3.2-1B and Llama-3.2-3B
```

Local runs are CPU-only by default, with 4 threads and an in-process abort at 6 GB of resident memory
(`--max-rss-gb`), because GPU runs on a laptop stalled the machine twice. On a cloud GPU pass
`--device cuda --dtype bfloat16 --threads 8 --max-rss-gb 40`. Llama-3.1-8B access is still pending,
so experiment 2 uses Llama-3.2-3B as the target.

## Terms

- **Chunk length L.** Each projection input of length d is split into d/L chunks of L numbers.
- **Centroids.** For each chunk position, 16 typical chunks found by k-means on calibration
  activations. Each input chunk is replaced by one of them, so it is stored as a 4-bit code.
- **Table.** For each chunk position and centroid, the centroid's precomputed product with the matching
  slice of the weight matrix, stored in 8 bits. A projection's output is the sum of one table row per
  chunk, so it reads d·m/L bytes per token against 2·d·m for 16-bit weights.
- **Encoder.** The rule that picks the centroid for an input chunk:
  - `nearest`: exact nearest centroid. The accuracy ceiling for a fixed set of centroids.
  - `tree`: the 4-level MADDNESS decision tree of Blalock and Guttag (2021).
  - `hash`: 4 random hyperplanes per chunk position, giving a 4-bit locality-sensitive hash.
- **Ridge refit.** After the centroids are fixed, table entries are re-solved by ridge regression
  against the exact layer output, shrinking toward the centroid products (`--lam`, default 3).

## Experiment 1: single-layer reconstruction

`exp1_single_layer.py` replaces one projection of Llama-3.2-1B (layer 8 of 16) and measures the damage
at the layer and at the model's output.

- **Projections.** `mlp.down_proj` (8192 → 2048, the widest input) and `self_attn.o_proj` (2048 → 2048).
- **Methods.** 8-bit weights (one scale per output column); 4-bit weights (one scale per 128 inputs per
  output column); tables at L of 8, 16 and 32, crossed with the three encoders and with
  SmoothQuant-style per-channel scaling on or off.
- **Data.** Calibration rows from WikiText-103 train; evaluation on WikiText-103 test, disjoint.
- **Metrics.** `rel_err` is the relative output error of the projection on held-out rows. `kl` is the
  mean KL divergence between the full model's next-token distributions with and without the
  replacement. `top1` is the fraction of positions where the two models' most likely token agrees.
- **Gate.** At L = 8 the best table's KL is within 2× that of 4-bit weights.

```
.venv/bin/python experiments/exp1_single_layer.py
.venv/bin/python experiments/exp1_single_layer.py --model Qwen/Qwen3-0.6B --layer 14   # ungated stand-in
```

`diag_layer.py` isolates the sources of layer error (table rounding, ridge fit, chunk length,
centroid count) without running the model.

### Results, Llama-3.2-1B layer 8 (`results/exp1_llama/`)

30K calibration rows, 8K evaluation tokens, per-channel scaling off.

| Method | down_proj KL | top-1 | o_proj KL | top-1 |
| --- | --- | --- | --- | --- |
| 8-bit weights | 0.00003 | 99.7% | 0.00002 | 99.8% |
| 4-bit weights | 0.0029 | 97.1% | 0.0022 | 97.4% |
| Tables L = 8, nearest | 0.062 | 86.8% | 0.036 | 89.8% |
| Tables L = 8, tree | 0.082 | 84.8% | 0.039 | 89.2% |
| Tables L = 8, hash | 0.104 | 83.2% | 0.058 | 87.4% |
| Tables L = 16, nearest | 0.083 | 84.4% | 0.047 | 88.1% |
| Tables L = 32, nearest | 0.104 | 82.8% | 0.059 | 87.2% |

- The best table at L = 8 costs about 20× the KL of 4-bit weights on the down-projection and 16× on
  the attention projection, so the gate fails.
- Encoder quality matters. The order nearest > tree > hash holds everywhere, and switching from hash
  to nearest is worth as much as reading 4× more bytes (L = 32 to L = 8).
- The attention projection tolerates tables better than the MLP projection.

### Diagnostic, Qwen3-0.6B layer 14 (`results/smoke_qwen/`, `results/diag_qwen_down.log`)

Relative output error of the down-projection on held-out rows:

| Method | Relative error |
| --- | --- |
| 4-bit / 3-bit / 2-bit weights | 0.14 / 0.33 / 0.81 |
| Tables L = 8: nearest / tree / hash | 0.58 / 0.67 / 0.71 |
| Tables L = 4, nearest | 0.48 |
| Tables L = 16, nearest | 0.67 |
| Tables L = 8, 256 centroids, nearest | 0.44 |

Rounding table entries to 8 bits adds no measurable error, and per-channel scaling does not help. The
error comes from representing each 8-number chunk by one of 16 centroids. More centroids or shorter
chunks reduce it, but neither comes close to 4-bit weights on their own.

## Experiment 2: acceptance under naive row choice

`exp2_acceptance.py` replaces projections across the whole drafter and measures how often the target
accepts its tokens. Drafter Llama-3.2-1B, target Llama-3.2-3B, one A10G (AWS g5.2xlarge).

- **Replaced sets.** `all` (q, k, v, o, gate, up and down of every block, 112 projections), `mlp`,
  `attn` or `down`, optionally restricted to a layer range (`down@4-15`). Embeddings and the output
  head stay exact. References: the exact 16-bit drafter and 4-bit weights.
- **Calibration.** 16K rows per projection input from 200K WikiText-103 train tokens, recorded from
  the exact drafter. With `--sequential`, rows are recorded layer by layer from the drafter with
  earlier layers already replaced, as in GPTQ.
- **Acceptance, measured two ways on WikiText-103 test.**
  - Teacher-forced on 16K tokens: α = E[Σ min(p_target, q_draft)], the per-token acceptance
    probability of speculative sampling.
  - 512 draft rounds: the drafter greedily writes 5 tokens, the target scores them in one pass, and
    the leading run of agreements is counted. This includes the drafter's errors compounding along its
    own continuation.
- **Gate.** The tree encoder at L = 8 lands within 0.1 of the exact drafter's α.

```
python experiments/exp2_acceptance.py --target meta-llama/Llama-3.2-3B --out results/exp2
python experiments/exp2_acceptance.py --sequential --cal-tokens 100000 --configs lut:nearest:8:down@4-15
python experiments/plot_exp2.py results/exp2_3b/acceptance.png results/exp2_3b/*/results.csv
```

### Results (`results/exp2_3b/`, plot `acceptance.png`)

| Drafter | α | Accepted tokens per 5-token round |
| --- | --- | --- |
| Exact 16-bit | 0.74 | 2.13 |
| 4-bit weights, all 112 projections | 0.66 | 1.73 |
| 4-bit weights, layers 4–15 | 0.70 | 1.88 |
| Tables, down-projection of one layer (2, 3 or 8) | 0.72 | 1.9–2.15 |
| Tables, down-projection of layer 0 alone, or layer 1 alone | 0.05 | 0.02 |
| Tables, down-projections of layers 4–15 | 0.09 | 0.25 |
| Tables, attention projections of layers 4–15 | 0.43 | 0.64 |
| Tables, all projections of layers 4–15 | 0.06 | 0.15 |
| Tables, every projection | 0.03 | 0.00 |
| Tables L = 16, down-projections of layers 4–15 | 0.17 | 0.42 |
| Tables, hash encoder, down-projections of layers 4–15 | 0.27 | 0.42 |
| Tables, tree encoder, down-projections of layers 4–15 | 0.07 | 0.19 |

Tables are L = 8 with the nearest encoder unless stated.

- **One table layer is cheap, many are not.** A single down-projection costs 2 points of α, matching
  experiment 1, but 12 of them bring α to 0.09 and all 112 bring it to 0.03. The damage compounds far
  faster than it adds.
- **Layers 0 and 1 must stay exact.** Either one alone collapses the drafter, while layers 2 and up
  behave like layer 8. These are the layers where Llama forms its very large activations, which an
  average over a centroid's bucket cannot reproduce.
- **Attention tolerates tables far better than the MLP.** The 48 attention projections of layers 4–15
  keep α at 0.43; the 36 MLP projections of the same layers give 0.06.
- **Calibration changes do not help.** Sequential calibration gives 0.096 against 0.088 on the
  down-projections of layers 4–15, 0.41 against 0.43 on attention, and 0.06 on everything. Stronger
  shrinkage (`--lam 30`) gives 0.081 and 0.31. The per-layer error itself is too large.
- **The single-layer encoder ranking does not survive compounding.** On the down-projections of
  layers 4–15, the hash encoder (0.27) and L = 16 (0.17) beat nearest at L = 8 (0.09). Coarser
  tables seem more robust to inputs already corrupted by earlier layers. One hash run with
  sequential calibration collapsed to 0.00, which is unexplained.
- **The gate fails.** Naive row choice with 16 centroids at L = 8 does not give a usable whole-model
  drafter.

Implications for the plan:

- Target-guided row choice (experiment 3) has to cut the per-layer error by a large factor for a
  whole-model table drafter to exist.
- The realistic operating point is a hybrid: tables on the attention projections of layers 2 and up,
  4-bit or exact weights for the MLPs and the first two blocks.
- Table fine-tuning against the target's distribution (experiment 7) is needed, not optional.

Fitting all 112 projections takes about 2 minutes with the nearest encoder and 8 with the tree on an
A10G. Evaluating one configuration takes 30 to 80 seconds.

## Experiment 4: fused kernel (first pass)

`bench_kernel.py` times a Triton kernel that reads each chunk's 4-bit code, gathers that 8-bit table
row and adds it into an int32 accumulator in one pass. A second version also computes the codes inside
the kernel with the hash encoder. Both are timed at batch 1 with CUDA graphs against cuBLAS BF16
matrix-vector products, on one A10G (AWS g5.xlarge). Tables and codes are random because the timing
depends only on shapes; the kernel output matches a PyTorch reference exactly.

```
python experiments/bench_kernel.py
```

### Results (`results/kernel_a10g.log`)

| Batch 1, A10G | BF16 dense | Tables, lookup only | Tables, with hash encoding |
| --- | --- | --- | --- |
| One down-projection (8192 → 2048) | 78 µs | 9.5 µs (8.2×) | 10.2 µs (7.6×) |
| All 112 projections of Llama-3.2-1B | 4.4 ms | 0.76 ms (5.8×) | 0.82 ms (5.3×) |

- **The byte savings turn into speed.** The table kernel reads 16× fewer bytes and runs 5.8× to 8.2×
  faster. It reaches about half the memory bandwidth of the dense product, so tuning has room left.
- **Encoding inside the kernel is cheap.** Computing the hash codes adds about 8%. The nearest and
  target-guided encoders cost more and are not yet timed.
- **Scope.** Projections only. Attention scores, normalization and the output head are excluded, and
  there is no 4-bit kernel comparison yet, which the proposal's gate asks for.
- A PyTorch gather-and-sum on a laptop CPU, without a custom kernel, was 2× slower than BF16. The
  speedup needs the fused kernel.

## Implementation notes

- On CUDA the table forward in `lut.py` turns codes into a one-hot matrix and multiplies it by the
  table in float32. This is 70× faster than `F.embedding_bag` and gives the same integer sums; TF32
  must stay off. It is for accuracy experiments only and reads more bytes than a dense layer.
- The ridge solve runs on the device that holds the codes.
