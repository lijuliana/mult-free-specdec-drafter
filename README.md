# Multiplication-Free Drafting for Speculative Decoding

CS 2420 Group 2 · Juliana Li, Eric Xu, Bowen Yang, Adam Zhou

A drafter for speculative decoding whose linear layers are precomputed lookup tables (MADDNESS,
Blalock and Guttag 2021), with table rows chosen using the target model's own features. At batch
size 1 the tables read 1/16 of the bytes of 16-bit weights, and the target still verifies every token,
so output quality is unchanged.

## Documents

- [PROPOSAL_SHORT.md](PROPOSAL_SHORT.md): the condensed proposal.
- [PROPOSAL.md](PROPOSAL.md): the full proposal, with the cost model, related work and references.
- [experiments/README.md](experiments/README.md): how each experiment is run, and its full results.

## Status

Llama-3.2-1B drafter, Llama-3.2-3B target (Llama-3.1-8B access pending), WikiText-103, one A10G. Tables
use 16 centroids per chunk of 8 inputs, with 8-bit entries. Acceptance is the per-token acceptance rate
against the target. Timing is one decode step at batch 1 with a 16-bit output head; projections without
tables run at 4 bits. In the acceptance runs, projections without tables stayed exact (16 bits).

| Drafter | ms per draft token | Acceptance needed to match 4-bit | Acceptance, best without target features | Acceptance with target features (same training without them) |
| --- | --- | --- | --- | --- |
| Exact 16-bit | 6.55 | 0.83 | 0.74 | |
| 4-bit weights everywhere | 3.45 | 0.66 | 0.66 | |
| Tables everywhere | 3.38 | 0.65 | 0.03 untrained | |
| Tables on attention, layers 2–15 | 3.63 | 0.67 | 0.44 | 0.42 (0.43) |
| Tables on MLP down-projections, layers 4–15 | about 3.38 (estimated) | about 0.65 | 0.45 | 0.35 (0.36) |
| Tables on all MLP projections, layers 4–15 | 3.24 | 0.64 | 0.06 untrained | not run |

What we know:

- **Speed.** Table projections are 5.8× faster than 16-bit ones, but a 4-bit drafter is nearly as fast,
  because attention, normalization, launch overhead and the output head dominate once weights are 4-bit.
  A table drafter step is 2% faster than a 4-bit one today, and about 30% faster in the best case
  (fused kernel, table output head).
- **Accuracy.** One table layer costs 2 points of acceptance, but the error compounds: tables in every
  projection collapse the drafter. Training the tables against the target recovers much of it for
  partial sets (0.09 → 0.45 on 12 down-projections), not for whole layers.
- **Target-guided row choice.** The first attempt made the drafter worse; the cause was an optimizer
  scaling bug, now fixed. With stable training the target's features tie a row choice trained without
  them. The main claim is not supported in this design.
- **What it takes to beat 4-bit.** A table drafter needs about 0.64 acceptance (0.59 in the best speed
  case) to match the 4-bit drafter's speedup at 0.66.

Open directions: bandwidth-bound hardware (phones, CPUs) where table bytes dominate, a table output
head, a 1–2 layer drafter fed by the target's features so errors compound through fewer layers, and
feeding the features into the drafter continuously instead of only through the row choice.

## Layout

```
PROPOSAL.md, PROPOSAL_SHORT.md   proposal
figures/                         draft-verify schematic (TikZ source and renders)
experiments/lut.py               table construction, encoders, ridge refit, table layer
experiments/exp1_single_layer.py experiment 1: one projection replaced, KL and top-1
experiments/diag_layer.py        experiment 1 diagnostic: sources of layer error
experiments/exp2_acceptance.py   experiment 2: whole drafter, acceptance against the target
experiments/exp7_finetune.py     experiment 7: train table entries against the target
experiments/exp3_guided.py       experiment 3: row-choice variants, with and without target features
experiments/bench_kernel.py      experiment 4: fused Triton table kernel against BF16
experiments/bench_drafter.py     experiment 4: whole drafter step, 16-bit vs 4-bit vs tables
experiments/plot_exp*.py         figures
results/                         CSVs, logs and plots for every run
```
