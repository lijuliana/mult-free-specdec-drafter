# Experiments

Code and results for the starter experiments of [PROPOSAL.md](../PROPOSAL.md). Raw outputs are in
[`results/`](../results).

| Experiment | Status | Headline |
| --- | --- | --- |
| 1. Single-layer reconstruction | Done (Llama-3.2-1B) | Best table at L = 8 costs about 20× the KL of 4-bit weights. Gate failed. |
| 2. Acceptance under naive row choice | Done (1B drafting for 3B) | One table layer costs 2 points of acceptance; tables in every layer collapse the drafter. Gate failed. |
| 7. Table fine-tuning | Done (partial sets) | Training rescues a lot: MLP down-projections of layers 4–15 go from 0.09 to 0.45 acceptance. |
| 3. Target-guided row choice | Done (simplified) | Target features add nothing over an equally trained row choice without them (0.42 vs 0.43). |
| 4. Fused kernel and drafter timing | Done (A10G) | Table projections are 5.8× faster than BF16, but a whole table drafter step is only 2% faster than a 4-bit one. |
| 5, 6 | Not started | |

**Where things stand.** Tables make each projection fast, but on a GPU a 4-bit drafter is already
nearly as fast once attention, normalization, launch overhead and the output head are counted. A table
drafter needs an acceptance of about 0.64 to beat the 4-bit drafter's 0.66. The best table drafters
measured so far reach 0.45 with 12 of 112 projections replaced, and about 0.16 with all projections of
layers 8–15. Feeding the target's features into the row choice did not help in this design. The
remaining options are bandwidth-bound hardware (phones, CPUs), a table output head, and a much
shallower drafter.

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

## Experiment 7: table fine-tuning

`exp7_finetune.py` starts from the experiment-2 tables, makes every table entry a trainable float, and
trains all of them jointly so the drafter's next-token distribution matches the 3B target's
(KL(p_target || q_draft) on 2M WikiText-103 training tokens). The encoders stay fixed. A table layer has
no gradient with respect to its input, so the backward pass uses the original weight matrix in its
place (straight-through). Trained tables are rounded back to 8 bits and evaluated as in experiment 2.

```
python experiments/exp7_finetune.py --configs lut:nearest:8:attn@2-15 lut:nearest:8:down@4-15
```

## Experiment 3: target-guided row choice

`exp3_guided.py` keeps the experiment-2 tables and changes how each chunk picks its row. Every variant
scores the 16 rows of a chunk and takes the best one:

- `recon`: the row whose stored answer is closest to the chunk's exact product. Uses only the input.
- `learned`: nearest-centroid score plus a learned linear term in the chunk, trained end to end against
  the target. The control: training without target features.
- `guided`: `learned` plus a term from the target's features. Hidden states of 5 evenly spaced target
  layers (as in DFlash) are normalized, concatenated and projected; a per-projection adapter maps the
  result to 32 numbers, and a per-chunk matrix turns those into a score for each row. Position t sees
  the target feature from position t − 1 (or t − 3), and draft rounds hold the feature of the last
  verified token.
- `joint`, `joint_guided`: as `learned` and `guided`, but centroids and table entries train too.
- `warmstart`: trains `learned`, then continues on new data twice from the same state, once without and
  once with the context term (which starts at zero). This is the clean test of the features.

Training uses the hard choice in the forward pass and a softmax over the scores for the gradient
(straight-through), 1M to 2M training tokens per run.

```
python experiments/exp3_guided.py --configs lut:nearest:8:attn@2-15 lut:nearest:8:down@4-15
python experiments/exp3_guided.py --variants warmstart --train-tokens 2000000 --train-batch 1
```

### Results, experiments 7 and 3 (`results/exp7_3b/`, `results/exp3_3b/`)

Acceptance per token against Llama-3.2-3B (teacher-forced E[min(p, q)]):

| Method | Attention, layers 2–15 | Down-projection, layers 4–15 |
| --- | --- | --- |
| Tables, nearest centroid (experiment 2) | 0.35 | 0.09 |
| `recon` row choice, untrained | 0.32 | 0.09 |
| Fine-tuned tables (experiment 7) | 0.44 | 0.37 |
| `learned` row choice, no target features | 0.43 | 0.36 |
| `joint`: tables, centroids and row choice trained together | 0.41 | **0.45** |
| `warmstart` control, no features | 0.43 | 0.36 |
| `warmstart` with target features | 0.42 | 0.35 |
| First `guided` run (broken optimization) | 0.31 | 0.18 |

Fine-tuning tables in every projection of layers 8–15 (Adafactor, to fit in 24 GB) raised acceptance
from 0.07 to 0.16. Training tables, centroids and row choice together (`joint`, 2M tokens, 8-bit Adam
on an L40S, `results/exp3_3b/all_tables/`) raised tables in every projection from 0.03 to 0.30, and
tables in every projection of layers 2–15 (layers 0–1 exact) from 0.06 to 0.33. The exact drafter is at 0.74, and 4-bit weights everywhere at 0.66.

- **Training rescues much of the compounding.** The down-projection set goes from 0.09 to 0.45 when
  tables, centroids and row choice train together. Tables in every projection remain far off.
- **A smarter untrained row choice does not help.** Picking the row that best matches each chunk's
  exact product is slightly worse than the nearest centroid, because the ridge-fit tables are tuned
  jointly across chunks.
- **The first guided runs got worse than their own starting point** (validation KL 2.23 → 2.77 on
  attention and 3.73 → 5.03 on down-projections within 200 steps), so the drop was an optimization
  failure. Adam moves each weight by about the learning rate, so a unit summing n inputs moves about
  n times faster: the adapter (1,024 inputs) and fusion projection (15,360 inputs) turned the
  zero-initialized context term into a growing random bias. Warm starting at a 10× lower learning rate
  did not fix it. A fixed random fusion projection, a unit-RMS context vector and an adapter learning
  rate divided by its fan-in did: the context term then stays small (up to 11% of the base score) and
  training is stable.
- **With stable training, target features tie the control.** The likely limits are the design (the
  chunk itself already determines its best row, and features enter only through a hard 16-way choice
  with a biased straight-through gradient) and the training scale (about 1,000× less data than
  DFlash, which feeds features continuously into every draft layer's keys and values).

## Experiment 4: fused kernel and drafter timing

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

### Whole drafter step (`bench_drafter.py`, `results/drafter_timing/`)

`bench_drafter.py` times one decode step of the whole 1B drafter at batch 1: RMSNorm, q/k/v, attention
over a 512-token cache, output projection, MLP and output head, captured in one CUDA graph. Table
projections use nearest-centroid encoding (a Triton kernel) and the fused gather-and-add. 4-bit
projections use PyTorch's tinygemm kernel (`torch._weight_int4pack_mm`, as in gpt-fast).

| Drafter, Llama-3.2-1B on an A10G | ms per token | Acceptance needed to match the 4-bit drafter |
| --- | --- | --- |
| 16-bit | 6.55 | 0.83 |
| 4-bit weights | 3.45 | 0.66 (measured) |
| Tables in every projection | 3.38 | 0.65 |
| Tables on attention (all layers), 4-bit MLP | 3.66 | 0.67 |
| Tables on attention 2–15, 4-bit elsewhere | 3.63 | 0.67 |
| Tables on MLP (all layers), 4-bit attention | 3.17 | 0.64 |
| Tables on MLP 4–15, 4-bit elsewhere | 3.24 | 0.64 |
| 4-bit weights, 4-bit output head | 2.64 | 0.60 |
| Tables on attention 2–15, 4-bit elsewhere, 4-bit output head | 2.82 | 0.61 |
| Tables on MLP, 4-bit attention, 4-bit output head | 2.35 | 0.57 |
| Tables in every projection, 4-bit output head | 2.56 | 0.59 |

Second run, `results/drafter_timing/timing_a10g_mlp_attn.log`; the first run (`timing_a10g.log`)
agrees within 0.5%. On an L40S (`timing_l40s.log`) everything is about 1.5× faster and the ordering is
the same, except that tables in every projection are 2% slower than 4-bit (2.37 against 2.33 ms);
tables in layers 2–15 with 4-bit layers 0–1 take 2.37 ms, and the 3B target 10.5 ms.

**Speed against accuracy, attention and MLP tables.** Timing puts 4-bit weights on every projection
without tables. The acceptance runs kept those projections exact, so with 4-bit weights there they
would be a few points lower.

| Tables on | ms per token | Needed to match 4-bit | Best acceptance without target features | Acceptance with target features | Same training without features |
| --- | --- | --- | --- | --- | --- |
| Attention, layers 2–15 | 3.63 | 0.67 | 0.44 (fine-tuned tables) | 0.42 | 0.43 |
| MLP down-projections, layers 4–15 | about 3.38 (estimated) | about 0.65 | 0.45 (tables, centroids and row choice trained) | 0.35 | 0.36 |
| All MLP projections, layers 4–15 | 3.24 | 0.64 | 0.06 (untrained) | not run | not run |

The down-projection-only drafter was not timed; its estimate subtracts 12 × (20.4 − 14.1) µs from the
4-bit drafter. The context path of the guided encoder (fusion projection per round, adapter and scorer
per projection) is not included in any timing. Attention tables are the more accurate but make the
drafter slower than 4-bit; MLP tables make it faster but are the least accurate before training.

The 3B target takes 16.2 ms per token and the 8B target 35.0 ms. Break-even uses the speedup formula
with γ = 5, against the 4-bit drafter at its measured acceptance of 0.66 (1.30×).

| Per projection, µs | 16-bit | 4-bit | Tables |
| --- | --- | --- | --- |
| q, o (2048 → 2048) | 20.1 | 6.4 | 8.2 |
| k, v (2048 → 512) | 6.2 | 3.5 | 7.1 |
| gate, up (2048 → 8192) | 70.3 | 19.4 | 13.9 |
| down (8192 → 2048) | 77.7 | 20.4 | 14.1 |
| Output head (2048 → 128256) | 1084 | 269 | 93 |

- **Everything except the projections costs 1.1 ms** (attention, normalization, launches), and the
  16-bit output head another 1.1 ms. After 4-bit quantization the projections are a third of the step,
  so reading 4× fewer bytes again saves little.
- **Tables win on wide shapes and lose on narrow ones.** They beat 4-bit on the MLP projections and the
  output head, and lose on the small attention projections, where launch latency dominates. Accuracy
  favors tables on attention; speed favors them on the MLP and the head.
- **Best case.** With a fused one-kernel table projection (about half the current time) and a table
  output head, a table drafter step would take about 1.8 ms against 2.6 ms for a 4-bit drafter with a
  4-bit head. It would still need an acceptance of about 0.59 to match the 4-bit drafter at 0.66.

## Implementation notes

- On CUDA the table forward in `lut.py` turns codes into a one-hot matrix and multiplies it by the
  table in float32. This is 70× faster than `F.embedding_bag` and gives the same integer sums; TF32
  must stay off. It is for accuracy experiments only and reads more bytes than a dense layer.
- The ridge solve runs on the device that holds the codes.
