# Multiplication-Free Self-Drafting for Speculative Decoding

Juliana Li · October 2026

## Summary

We propose a drafter for speculative decoding that runs the target model's own layers with table lookups in place of multiplications. Each linear layer's input activation is split into short subvectors, each subvector is assigned to one of 16 learned prototypes, and the dot product of every prototype with the matching slice of the weight matrix is precomputed into a lookup table. This is the MADDNESS construction of [Blalock and Guttag (2021)](https://arxiv.org/abs/2106.10860). A drafter forward pass then reads tables and adds 8-bit entries, performs no multiply-adds, and at batch size 1 reads 1/16 of the bytes the target reads at the default subvector length of 8. The target verifies drafted tokens exactly as in standard speculative decoding, so output quality is unchanged.

Two properties make drafting the right place for this approximation. A drafter only has to propose tokens the target would accept, so the approximation error that keeps MADDNESS out of exact inference lowers the acceptance rate and leaves correctness untouched. The verification pass also computes the target's exact hidden states for every accepted position, and those states can condition the drafter's prototype assignment, which is the step whose latency limits the drafter on a GPU.

The first milestone is a measured acceptance rate and draft latency for a MADDNESS-compressed Llama-class model at 1B to 8B parameters, against two baselines: a 4-bit weight-quantized self-drafter and layer-skipping self-speculation.

## Motivation

Draft cost is the lever that current self-drafters leave unused. Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)) has a cheap drafter propose γ tokens, and the target checks them in one parallel pass. With acceptance rate α and drafter cost c per token (as a fraction of one target forward pass), the speedup per round is

$$
\text{speedup} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)\,(\gamma c + 1)}
$$

Both α and c enter, but two measurement studies find that c decides. [Ha et al. (ICML 2026)](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf) characterize five self-speculative methods on Llama-2 7B to 70B and state their first takeaway as "Draft cost, not acceptance rate, dominates throughput": the methods with the smallest drafters (Kangaroo, DEL) consistently beat the methods with better drafts (SWIFT, Draft & Verify). [Yan et al. (NAACL 2025)](https://arxiv.org/abs/2402.01528) reach the same conclusion for separate draft models: performance "depends heavily on the latency of the draft model", and the drafter's language-modeling quality "does not correlate strongly" with its speculative performance.

Existing self-drafters lower c by running fewer layers, either by skipping layers ([Draft & Verify](https://aclanthology.org/2024.acl-long.607); SWIFT) or by exiting early ([LayerSkip](https://aclanthology.org/2024.acl-long.681); [Kangaroo](https://arxiv.org/abs/2404.18911); DEL). The layers that remain run at full precision, so c is tied to depth, and depth is a coarse knob: Ha et al. report early-exit drafters using 7% to 25% of the layers and layer-skipping drafters 47% to 61%, with acceptance falling as layers are removed. Quantized self-drafters ([QSpec](https://arxiv.org/abs/2410.11305); [QuantSpec](https://arxiv.org/abs/2502.10424)) lower per-layer cost instead, with 4-bit weights and activations, and QuantSpec reports acceptance above 90%. A 4-bit matrix-vector product on a GPU is already close to the memory-bandwidth floor, so lowering c further needs a different representation of the layer.

MADDNESS is such a representation. For a weight matrix known in advance it replaces the product with table lookups and 8-bit additions and performs zero multiply-adds; [Blalock and Guttag (2021)](https://arxiv.org/abs/2106.10860) report up to 100× over exact products on CPU. It was never adopted for exact inference because its error compounds through depth: [McCarter and Dronen (2022)](https://arxiv.org/abs/2207.05808) applied it to whole networks with fine-tuning and found accuracy "substantially diminished compared to exact matrix multiplication". A drafter is the one place where that failure is affordable. The target verifies every drafted token, so approximation error lowers α and leaves output quality unchanged, and the cost model above says α is the cheaper currency.

Speculative decoding also gives the drafter something exact inference never has: the target's exact hidden states at every verified position, computed during verification. The drafter's data-dependent step is assigning each activation subvector to a prototype. Conditioning that assignment on the target's exact feature at the preceding verified position should let a smaller encoder, with fewer codebooks, reach the same acceptance rate, and testing this is the second aim of the project.

## Background and related work

Four lines of work meet here, and none of them has built a drafter from lookup tables. The table lists the methods this proposal is measured against and what each one changes about the drafter.

| Method | Drafter | How draft cost is cut | Extra training | Relation to this proposal |
| --- | --- | --- | --- | --- |
| [Draft & Verify](https://aclanthology.org/2024.acl-long.607) (ACL 2024) | Target with about half its layers skipped | Depth | None | Baseline. Draft cost 49% to 57% of the target per step (Ha et al.) |
| [LayerSkip](https://aclanthology.org/2024.acl-long.681) (ACL 2024) | Target's first 7 to 8 layers | Depth | Retrain the full model | Baseline. Draft cost 10% to 32% |
| [Kangaroo](https://arxiv.org/abs/2404.18911) (2024) | Target's first 2 to 6 layers plus a small adapter | Depth | Adapter only | The cost bar to beat: 8% to 13% |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) (NeurIPS 2025) | One trained layer fed the target's features | Separate small drafter | Train the drafter on a large corpus | We reuse its feature conditioning without training a new transformer layer |
| [QSpec](https://arxiv.org/abs/2410.11305) (EMNLP 2025) | Target at 4-bit weights and activations | Word size | None | Nearest baseline: same shared-weight design, with bits instead of tables |
| [QuantSpec](https://arxiv.org/abs/2502.10424) (ICML 2025) | Target at 4-bit weights with a 4-bit KV cache | Word size, KV bytes | None | Long-context baseline; acceptance above 90% |
| [MADDNESS](https://arxiv.org/abs/2106.10860), [LUT-NN](https://arxiv.org/abs/2302.03213), [Stella Nera](https://arxiv.org/abs/2311.10207) | Whole network through tables (not a drafter) | Multiplies become lookups | Centroid fine-tuning | The mechanism. Never applied to a decoder LLM or to drafting |

**Speculative decoding.** [Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192) and [Chen et al. (2023)](https://arxiv.org/abs/2302.01318) introduced the draft-then-verify loop with a rejection rule that preserves the target's output distribution. [Medusa](https://arxiv.org/abs/2401.10774) and [EAGLE](https://arxiv.org/abs/2401.15077) (ICML 2024) replaced the separate drafter with heads trained on the target's own hidden states; EAGLE's drafter takes the target's second-to-top-layer feature at the previous position plus the next token's embedding, and EAGLE-3 reaches speedups up to 6.5× with a trained one-layer drafter fed features from several target layers. These trained drafters set the speed bar, at the price of training a new component on a large corpus.

**Self-drafting.** Draft & Verify, LayerSkip, Kangaroo, SWIFT and DEL draft with a subset of the target's layers. [Ha et al. (ICML 2026)](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf) measured all five on Llama-2 7B to 70B and found that the methods with the cheapest drafters win despite lower acceptance, and that early-exit methods also win on verification because the target reuses the draft layers' computation. [Yan et al. (NAACL 2025)](https://arxiv.org/abs/2402.01528) found the same for separate drafters.

**Lookup-based matrix products.** MADDNESS descends from product quantization ([Jégou et al., 2011](https://ieeexplore.ieee.org/document/5432202)): it splits each input row into C subvectors, assigns each to one of 16 prototypes with a four-level decision tree, and sums 8-bit table entries that hold each prototype's dot product with the weights. [LUT-NN](https://arxiv.org/abs/2302.03213) (MobiCom 2023) made the prototypes differentiable and ran whole networks on tables with up to 16× fewer FLOPs. [Stella Nera](https://arxiv.org/abs/2311.10207) built the accelerator (161 TOp/s/W) and fine-tuned through the lookups. [McCarter and Dronen (2022)](https://arxiv.org/abs/2207.05808) tried whole-network inference and found accuracy "substantially diminished". The LUT kernels now used for LLMs ([LUT-GEMM](https://arxiv.org/abs/2206.09557), ICLR 2024; [T-MAC](https://arxiv.org/abs/2407.00088); [FLUTE](https://aclanthology.org/2024.findings-emnlp.724), EMNLP 2024) index tables by weight bit patterns and build the tables from activations at run time, so they are exact for a quantized weight. MADDNESS indexes by activation codes, so its tables are fixed and precomputed, and its error lives in the activations. We found no published application of activation-indexed tables to a decoder language model at or above 1B parameters.

**Quantized self-drafters.** QSpec drafts with 4-bit weights and activations and verifies with 4-bit weights at 16-bit activations, sharing weights and KV cache, for up to 1.64×. QuantSpec adds a hierarchical 4-bit KV cache for long contexts, reaching about 2.5×. Both show that a drafter can share the target's weights and KV cache with no retraining, which this proposal assumes.

## Proposed method

The drafter is the target model with every linear projection replaced by a lookup table; it shares the target's embeddings, normalization layers, attention arithmetic and KV cache, and the outer loop is standard speculative decoding.

![One draft-verify round: the table drafter encodes, looks up and adds through every layer; the target verifies in one pass; accepted positions return exact keys, values and a feature that conditions the next round's encoder.](figures/draft_verify_round.png)

Each round the drafter hashes, looks up and adds through every layer, the target verifies the drafted tokens in one pass, and the accepted positions return exact keys, values and a feature that conditions the next round's encoder.

**Table construction (offline, per projection).** A projection with weight W of shape d × m is split along its input dimension into C = d/L subvectors of length L. On about 100K activation rows collected from the target on 1M to 2M tokens of generic text, we fit K = 16 prototypes per subspace. Each table entry T_c[k, j] holds the dot product of prototype k in subspace c with the matching slice of column j of W, scaled to 8 bits as in MADDNESS. After the prototypes are fixed, the table entries are refit by ridge regression against the layer's exact outputs on the calibration set, which is closed-form and recovers part of the quantization error. Storage is 16·d·m/L bytes: equal to 8-bit weights at L = 16, 4-bit weights at L = 32 and 2-bit weights at L = 64.

**Encoder.** The encoder maps each activation subvector to a prototype index. MADDNESS uses a four-level decision tree per subspace (four scalar comparisons, no multiplies). The alternative is nearest-prototype assignment, which costs 16·L multiplies per subvector; this is small next to the d·m multiplies of the exact layer, but it is a sequence of data-dependent steps that run poorly on a GPU at batch size 1, which is the cost our TF flagged. Both encoders are implemented and timed in the first experiments.

**Draft forward pass.** Per token and layer, the drafter encodes the input to C codes and, for each of the m outputs, sums C 8-bit table entries. The Q, K, V, O projections and both MLP projections go through tables. Attention scores, softmax, RMSNorm, SiLU and residual additions stay exact in 16 bits; they are elementwise or scale with context length rather than with d·m. The LM head is the largest single matrix and is tested both as a table and as an exact 8-bit product.

**Sharing with the target.** Keys and values for every verified position come from the target's exact pass, as in QSpec and QuantSpec. The drafter writes its own approximate keys and values for draft positions into a scratch region that is dropped after verification. The target then runs one forward pass over the γ drafted tokens and applies the rejection rule of Leviathan et al., which keeps the output distribution exactly that of the target.

**Feature-guided encoding.** After verification the target's exact second-to-top-layer hidden state f_t at the last accepted position t is free. A one- or two-layer MLP a_l maps f_t to a 32-dimensional context vector for layer l, computed once per draft round for all layers. The guided encoder for subspace c of layer l scores the 16 prototypes from the concatenation of the activation subvector and that context vector, and takes the argmax. It is trained by cross-entropy against the prototype whose table entry best reconstructs the exact layer output on the calibration set, with a second variant trained end to end on the drafter's next-token log-likelihood under the target. The hypothesis is that context resolves subspace ambiguity, so the same acceptance rate is reached with fewer codebooks (longer L), which cuts both table bytes and encoder steps. For draft positions beyond the first, the context vector is either held at the last verified position (exact but stale) or recomputed from the drafter's own approximate feature (fresh but noisy); both are tested. When the drafter is the target itself, the adapter only has to map the top-layer feature space into each layer's encoder space. If a smaller model from the same family is used as the drafter instead, the same adapter bridges the two models' feature spaces, which is why the drafter must come from the target's family.

**Layer-selective compression.** The subvector length L is chosen per projection and per layer under a total-bytes budget. Sensitivity is measured as the increase in KL divergence between the drafter's and the target's next-token distributions when only that projection is replaced by a table. A greedy allocation then lengthens L where the KL cost per byte saved is lowest, in the same spirit as the Fisher-based rank allocation of [Palu](https://arxiv.org/abs/2407.21118) (ICLR 2025). Published results disagree on which attention projection tolerates compression best: Palu allocates higher rank to values than to keys under low-rank KV compression, and [Thin Keys, Full Values](https://arxiv.org/abs/2603.04427) (2026) finds keys tolerate SVD far better than queries. The allocation is therefore measured for this compression rather than assumed.

**What is not changed.** No target weight is modified, no new transformer layer is trained, and the output distribution is the target's. The trained components are the prototypes, the tables, and the small encoder MLPs.

## Novelty

The proposal's claim to novelty rests on four points, each stated against the closest prior work.

- **The first activation-indexed lookup-table drafter.** Every existing self-drafter cuts cost by depth (Draft & Verify, LayerSkip, Kangaroo, SWIFT, DEL) or by word size (QSpec, QuantSpec). This drafter changes the arithmetic itself: its projections perform no multiply-adds, and its error lives in the activations rather than the weights. The LUT kernels now used for LLMs (LUT-GEMM, T-MAC, FLUTE) are exact kernels for quantized weights and are a different object.
- **A setting where MADDNESS's known failure is affordable.** Whole-network lookup inference was tried and found to lose too much accuracy (McCarter and Dronen, 2022; Stella Nera recovers it only with fine-tuning on CIFAR-scale models). Speculative verification turns that accuracy loss into a lower acceptance rate with no change to outputs, and the measurements of Ha et al. and Yan et al. say acceptance rate is the cheaper currency. To our knowledge no one has made this argument or tested it.
- **Feature-guided encoding.** EAGLE showed that the target's hidden states at the previous position carry most of what a drafter needs. We use that signal to condition the prototype assignment inside each layer rather than to drive a new trained transformer layer. The trained components are per-layer MLP adapters and per-subspace scorers, a few megabytes in total, and they are trained on a calibration set rather than a corpus. This is also, as far as we can find, the first use of a verifier's exact features to steer a product-quantization encoder.
- **A continuous, per-projection cost knob.** Depth-based drafters choose which whole layers to run; quantized drafters choose among 2, 3, 4 or 8 bits. The subvector length L sets table bytes per projection on a continuous scale, and a KL-based allocation under a byte budget decides where the drafter can afford to be coarse. This yields the per-projection sensitivity map our TF suggested as an additional contribution.

A fifth point is a consequence rather than a claim: the tables and codes are the representation that LUT accelerators (Stella Nera) and CPU LUT kernels (T-MAC) already execute, so a drafter that works on a GPU has a hardware path that bit-quantized drafters do not.

## Feasibility and cost model

At batch size 1 the drafter reads 1/(2L) of the bytes the target reads, because each token selects one 8-bit row per codebook out of 16, and that is the regime where speculative decoding is used. Decoding at batch size 1 is bound by memory bandwidth, so per-token cost is well approximated by bytes read. A projection of shape d × m in 16-bit weights reads 2dm bytes per token. Its table holds C = d/L codebooks of 16 rows of m bytes, but a token reads only the one row its code selects in each codebook: C·m = dm/L bytes. The table is stored in full, so memory footprint and bytes read diverge, and the gap closes as batch size grows, because different tokens select different rows. At 16 or more tokens per batch the whole table is read.

| Subvector length L | Table storage, as a fraction of 16-bit weights | Bytes read per token at batch 1, as a fraction of 16-bit weights | Equivalent weight precision by storage |
| --- | --- | --- | --- |
| 8 (MADDNESS default) | 1.0 | 1/16 | 16-bit |
| 16 | 0.5 | 1/32 | 8-bit |
| 32 | 0.25 | 1/64 | 4-bit |
| 64 | 0.125 | 1/128 | 2-bit |

For Llama-3.1-8B (16 GB of weights in 16 bits), L = 8 means 16 GB of tables beside the target, which fits an 80 GB GPU and not a 24 GB one, and about 1 GB read per drafted token plus the LM head. A 4-bit weight-only drafter reads 4 GB per token. Against a target that is itself served at 4-bit weights the drafter's advantage shrinks by 4×, and L = 32 or an early-exit combination is needed to keep c below 0.1.

The speedup formula from the Motivation gives the targets the first experiments must hit. With γ = 5 drafted tokens per round:

| Drafter cost c | α = 0.7 | α = 0.8 | α = 0.9 |
| --- | --- | --- | --- |
| 0.50 (Draft & Verify, L = 16 at full-table reads) | 0.84× | 1.05× | 1.34× |
| 0.25 (4-bit drafter) | 1.31× | 1.64× | 2.08× |
| 0.10 (Kangaroo) | 1.96× | 2.46× | 3.12× |
| 0.05 (tables at L = 8 plus overheads) | 2.35× | 2.95× | 3.75× |

The model assumes verification of γ + 1 tokens costs one target forward pass, which holds at batch size 1, and it omits the encoder, attention over the KV cache, and kernel launch latency. The project is worthwhile if the measured c lands at or below 0.1 with α at or above 0.8, which the table puts at about 2.5× and ahead of every training-free self-drafter in Ha et al.

Four things can break the model, and the first experiments are designed around them.

- **Encoder latency.** The tree encoder costs four comparisons per subvector, which is nothing, but as a separate kernel it is a chain of dependent steps per projection: 6 projections × 32 layers × 4 levels is about 800 dependent launches per drafted token, and at a few microseconds each that approaches a full target forward pass on an H100. The encoder must be fused into the gather-and-add kernel of each projection, or replaced by the single small dense product of the feature-guided encoder. This is the bottleneck our TF named, and it is a latency problem rather than an arithmetic one.
- **Activation outliers.** LLM activations have a few channels with values 10× to 100× the rest, which is why activation quantization is harder than weight quantization. K-means on raw subvectors would spend its 16 prototypes on those channels. Per-channel scaling of the kind used by SmoothQuant folds into the weight before the table is built, so it is free here; whether it is enough is measured in the first experiment.
- **No reuse at verification.** Early-exit drafters let the target reuse the draft layers' computation; Ha et al. identify this as part of why they win. A table drafter's activations are approximate, so the target runs all its layers from scratch. The drafter must earn its place on draft cost alone.
- **Memory footprint.** The tables are extra memory beside the target. At L = 8 they double the footprint. A deployment with tight memory would use L = 16 or 32, or compress only the layers the sensitivity map allows.

## Further explorations

The drafter is cheap enough that several directions open once the first results are in. The first group extends the method; the second uses the artifact to answer questions about the target model.

**Extensions**

- **Tables plus early exit.** Build tables for the first E layers only and exit with a Kangaroo-style adapter. Depth and L become two knobs under one byte budget, and a layer that layer-skipping would drop can instead be kept at L = 64 for 1/128 of its cost, which may retain more acceptance than dropping it.
- **Table fine-tuning by distillation.** LUT-NN and Stella Nera make the lookups differentiable. Here the tables and prototypes are the only trained parameters, so fine-tuning them to match the target's next-token distribution on a few million tokens is cheap and attacks α directly.
- **Learned subspaces and rotations.** MADDNESS groups adjacent input dimensions. Choosing the grouping (as ITLUMM did) or applying an orthogonal rotation of the input, as QuaRot-style quantization does to remove outliers, folds into the weight before the table is built and costs nothing at run time.
- **Tree drafting.** With c near 0.05 the drafter can afford to propose a tree of candidates as EAGLE-2 does instead of a chain. Tree width raises the number of table rows read per round, so the table-read advantage fades around 16 candidates; the experiment finds the width that maximizes accepted tokens per second.
- **Batched serving.** At batch sizes of 8 to 64 the drafter reads most of its tables and becomes bound by 8-bit additions rather than bytes. Whether it still beats a 4-bit drafter there is a measurement, and it decides whether the method is a latency tool or also a throughput tool.
- **Long context.** The drafter shares the target's KV cache, which dominates bytes at 32K tokens and beyond. QuantSpec's 4-bit hierarchical cache, or a product-quantized cache, composes with table projections; the two have not been combined.
- **Hardware.** The tables and codes are what Stella Nera and T-MAC execute. A drafter that works on a GPU gives a direct target for a LUT accelerator or a CPU-side drafter on edge devices, where the multiply-free property matters most.

**Questions the artifact can answer**

- **A per-projection compressibility map.** Acceptance rate as a function of L for each projection and layer measures how many distinct input patterns each projection needs for next-token prediction. This is the result our TF suggested, and it can settle for this compression type whether values or keys tolerate more compression.
- **Temporal predictability of activations.** The feature-guided encoder's accuracy at predicting layer-l codes from the previous position's top-layer feature measures, per layer, how much of the current activation is already determined by context. EAGLE's finding that top-layer features are predictable is one point on this curve.
- **Error compounding through depth.** Comparing a drafter with tables in all layers to one with tables in half of them, at matched bytes, measures how approximation error accumulates in a decoder, which is the quantity McCarter and Dronen found prohibitive for exact inference.

## Starter experiments

The first three experiments take about three weeks on one 80 GB GPU and decide whether the project continues; the rest build the paper. Models are Llama-3.2-1B for iteration and Llama-3.1-8B or Llama-2-7B for comparison with Ha et al. Calibration uses 1M to 2M tokens of FineWeb or WikiText; evaluation uses CNN/DailyMail, GSM8K and HumanEval prompts, the three datasets of Ha et al.

1. **Single-layer reconstruction (week 1).** For one middle-layer MLP down-projection and one attention output projection, fit MADDNESS at L of 8, 16 and 32, with and without SmoothQuant-style per-channel scaling, and refit the tables by ridge regression. Measure relative output error and the KL divergence of the next-token distribution when only that layer is replaced. Compare with 8-bit and 4-bit weight quantization of the same layer. Pure PyTorch gather-and-sum, no kernels. Gate: at L = 8 the KL cost is within 2× that of 4-bit weights; otherwise outlier handling comes first.
2. **Whole-model acceptance rate (weeks 2 to 3).** Replace every projection at uniform L and measure the acceptance rate α with γ = 5 under greedy and sampled verification. Compare with the same model at 4-bit weights as drafter. Gate: α at or above 0.7 at L = 8 on the 1B model. Below that, table fine-tuning (experiment 7) moves ahead of everything else.
3. **Fused kernel microbenchmark (weeks 2 to 3, in parallel).** Write one Triton kernel per projection that encodes with the tree, gathers the selected rows and adds them in 8 bits, at batch size 1 on shapes 4096 × 4096 and 4096 × 14336. Time it against a BF16 matrix-vector product and a Marlin 4-bit kernel; report bytes read and achieved bandwidth. Gate: per-projection time at or below 0.15 of BF16 at L = 8, which is within 2.5× of the bandwidth model. Above that, fusion and launch overhead are the work, not the method.
4. **End-to-end speedup (week 4).** Run the drafter and target in one loop with a shared KV cache and CUDA graphs. Report c, α and tokens per second against vanilla decoding, Draft & Verify and LayerSkip from their released code, and QSpec if its code runs, on Llama-2-7B to match Ha et al.
5. **Sensitivity map and allocation (weeks 4 to 5).** Measure the KL cost of replacing each projection in each layer at L of 16, 32 and 64. Run the greedy allocation under byte budgets of 0.25× and 0.125× of the 16-bit weights and compare α with uniform L at the same budget. Report the map by projection type (Q, K, V, O, gate, up, down) and depth.
6. **Feature-guided encoder (weeks 5 to 6).** Train the per-layer adapters and per-subspace scorers on the calibration set against teacher codes. Measure code accuracy and α at L of 16 and 32 against the tree encoder at L = 8, with the context vector held at the last verified position and with it recomputed from the drafter's own feature. Time the encoder fused into the kernel from experiment 3.
7. **Table fine-tuning by distillation (if needed).** Fine-tune tables and prototypes through a straight-through estimator on 5M tokens against the target's next-token distribution and report the gain in α.

## Open questions

- **Self-drafter or small same-family drafter.** This proposal makes the drafter the target itself with tables, which shares the KV cache and needs no second set of weights. Our professor's framing of a same-family model with an adapter also fits a smaller model (Llama-3.2-1B drafting for 8B) with tables. If experiment 2 gives α below 0.7 for the 8B self-drafter at L = 8, the 1B variant is the fallback, and the adapter then bridges two models' feature spaces.
- **How much do 16 prototypes per 8-dimensional subspace capture of LLM activations?** There is no data point at 1B parameters or above. Outlier channels are the known obstacle, and whether per-channel scaling folded into the tables is enough is the first thing experiment 1 answers.
- **Can the encoder be fused on a GPU?** A tree with data-dependent branches inside a kernel causes warp divergence. Nearest-prototype assignment costs 16·L multiplies per subvector but is branch-free, and it may be the faster encoder on a GPU despite the multiplies. The multiply-free property matters for hardware. The GPU result does not depend on it.
- **Does context reduce the codebooks needed?** The feature-guided encoder rests on the hypothesis that the previous position's exact feature resolves subspace ambiguity. If code accuracy improves but α does not, the encoder is predicting codes that do not matter, and the training target should switch to the drafter's next-token likelihood.
- **Stale or fresh context beyond the first draft token.** Holding f_t at the last verified position is exact but ages by one token per draft step; recomputing it from the drafter's approximate feature is fresh but noisy. Which wins may depend on γ.
- **Sampled verification.** Under sampling, acceptance depends on the drafter's full distribution, and a table drafter's logits are distorted in ways that greedy acceptance hides. Temperature 0 and 1 are both measured.
- **Which target precision is the baseline?** Ha et al. use 16-bit targets. Against a target served at 4-bit weights the drafter's byte advantage drops by 4×, and the right comparison for a paper is both.
- **Which prior result on value compressibility did our TF mean?** We could not locate a 2025 paper showing values are more compressible, and the nearest verified results point in different directions (Palu gives values higher rank than keys; Thin Keys, Full Values finds keys compress better than queries). We need the reference, and experiment 5 measures the answer for this compression regardless.
- **Reporting.** Ha et al. recommend reporting draft cost and verification cost beside acceptance rate. Every result table in this project reports c, α and tokens per second together.

## References

Every entry was checked against its arXiv or venue page on 2026-10-26. Where a venue could not be confirmed the arXiv listing is given.

- Ashkboos, S. et al. (2024). QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Blalock, D. and Guttag, J. (2021). Multiplying Matrices Without Multiplying. ICML 2021, PMLR 139. [arXiv:2106.10860](https://arxiv.org/abs/2106.10860)
- Cai, T. et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Chang, C.-C. et al. (2025). Palu: KV-Cache Compression with Low-Rank Projection. ICLR 2025. [arXiv:2407.21118](https://arxiv.org/abs/2407.21118)
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L. and Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Egiazarian, V. et al. (2024). Extreme Compression of Large Language Models via Additive Quantization. ICML 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118)
- Elhoushi, M. et al. (2024). LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.681)
- Guo, H. et al. (2024). Fast Matrix Multiplications for Lookup Table-Quantized LLMs. Findings of EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.724)
- Ha, J., Ganesan, K., Nguyen, A., Ahmed, T. and Moshovos, A. (2026). Characterizing Self-Speculative Decoding Approaches for Accelerating LLMs. ICML 2026, PMLR 306. [PDF](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf)
- Jégou, H., Douze, M. and Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. IEEE TPAMI 33(1). [IEEE](https://ieeexplore.ieee.org/document/5432202)
- Leviathan, Y., Kalman, M. and Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023, PMLR 202. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Li, Y., Wei, F., Zhang, C. and Zhang, H. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. ICML 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Li, Y., Wei, F., Zhang, C. and Zhang, H. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test. NeurIPS 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- Liu, F. et al. (2024). Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting. NeurIPS 2024. [arXiv:2404.18911](https://arxiv.org/abs/2404.18911)
- McCarter, C. and Dronen, N. (2022). Look-ups are not (yet) all you need for deep learning inference. [arXiv:2207.05808](https://arxiv.org/abs/2207.05808)
- Park, G. et al. (2024). LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models. ICLR 2024. [arXiv:2206.09557](https://arxiv.org/abs/2206.09557)
- Schönleber, J. et al. (2023). Stella Nera: A Differentiable Maddness-Based Hardware Accelerator for Efficient Approximate Matrix Multiplication. ISVLSI 2025. [arXiv:2311.10207](https://arxiv.org/abs/2311.10207)
- Tang, X. et al. (2023). LUT-NN: Empower Efficient Neural Network Inference with Centroid Learning and Table Lookup. MobiCom 2023. [arXiv:2302.03213](https://arxiv.org/abs/2302.03213)
- Thin Keys, Full Values: Reducing KV Cache via Low-Dimensional Attention Selection (2026). [arXiv:2603.04427](https://arxiv.org/abs/2603.04427)
- Tiwari, R. et al. (2025). QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache. ICML 2025, PMLR 267. [arXiv:2502.10424](https://arxiv.org/abs/2502.10424)
- Wei, J. et al. (2024). T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge. [arXiv:2407.00088](https://arxiv.org/abs/2407.00088)
- Xiao, G. et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- Yan, M., Agarwal, S. and Venkataraman, S. (2025). Decoding Speculative Decoding. NAACL 2025. [arXiv:2402.01528](https://arxiv.org/abs/2402.01528)
- Zhang, J. et al. (2024). Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding. ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.607)
- Zhao, J., Lu, W., Wang, S., Kong, L. and Wu, C. (2025). QSpec: Speculative Decoding with Complementary Quantization Schemes. EMNLP 2025. [arXiv:2410.11305](https://arxiv.org/abs/2410.11305)

Named in the text without their own entry, as characterized by Ha et al. (2026): SWIFT (Xia et al., 2025) and DEL (Zarch et al., 2025).
