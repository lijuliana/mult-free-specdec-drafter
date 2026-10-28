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

## Results so far

Llama-3.2-1B drafter, Llama-3.2-3B target, WikiText-103. Tables use 16 centroids per chunk of 8 inputs.

| Result | Measurement |
| --- | --- |
| Fused table kernel against BF16, all 112 projections, batch 1, A10G | 0.76 ms against 4.4 ms (5.8×) |
| Acceptance, exact drafter / one MLP layer as tables | 0.74 / 0.72 |
| Acceptance, attention projections of layers 4–15 as tables | 0.43 |
| Acceptance, every projection as tables | 0.03 |

Tables are fast and one layer is cheap, but the error compounds through depth. The next steps are
target-guided row choice, table fine-tuning, and a hybrid drafter with tables on attention and 4-bit
MLPs.

## Layout

```
PROPOSAL.md, PROPOSAL_SHORT.md   proposal
figures/                         draft-verify schematic (TikZ source and renders)
experiments/lut.py               table construction, encoders, ridge refit, table layer
experiments/exp1_single_layer.py experiment 1: one projection replaced, KL and top-1
experiments/diag_layer.py        experiment 1 diagnostic: sources of layer error
experiments/exp2_acceptance.py   experiment 2: whole drafter, acceptance against the target
experiments/bench_kernel.py      experiment 4: fused Triton table kernel against BF16
experiments/plot_exp*.py         figures
results/                         CSVs, logs and plots for every run
```
