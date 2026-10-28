# Multiplication-Free Drafting for Speculative Decoding

CS 2420 Group 2 · Juliana Li, Eric Xu, Bowen Yang, Adam Zhou · October 2026

## Summary

We propose a drafter for speculative decoding whose linear layers run as table lookups in place of multiplications, and whose table indices are chosen with help from the target model's features. The drafter is a smaller model from the target's family, for example Llama-3.2-1B drafting for Llama-3.1-8B. Each linear layer's input activation is split into short subvectors, each subvector is assigned to one of 16 learned prototypes, and the dot product of every prototype with the matching slice of the weight matrix is precomputed into a lookup table. This is the MADDNESS construction of [Blalock and Guttag (2021)](https://arxiv.org/abs/2106.10860). A draft forward pass reads one table row per codebook and adds 8-bit entries, performs no multiply-adds in its projections, and at batch size 1 reads 1/16 of the bytes the same model reads in 16 bits at the default subvector length of 8. The target verifies drafted tokens exactly as in standard speculative decoding, so output quality is unchanged.

The main claim is about accuracy. Assigning each activation to a prototype is the drafter's one data-dependent step. The naive assignment hashes the activation to a bucket with a locality-sensitive hash and uses that bucket's centroid. We claim that conditioning the assignment on features the target computed during verification gives a higher acceptance rate than naive hashing at the same table budget. The target's features already encode several future tokens, which is what makes [EAGLE-3](https://arxiv.org/abs/2503.01840) and [DFlash](https://arxiv.org/abs/2602.06036) fast, and a drafter's errors only lower its acceptance rate, which is what makes MADDNESS's approximation affordable here.

The first milestone is a curve of acceptance rate against number of codebooks for Llama-3.2-1B drafting for Llama-3.1-8B, once with naive hashing and once with target-feature conditioning. Draft latency is measured beside it. The accuracy claim does not depend on the latency result.

## Motivation

Draft cost is the lever that current drafters leave unused. Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)) has a cheap drafter propose γ tokens, and the target checks them in one parallel pass. With acceptance rate α and drafter cost c per token (as a fraction of one target forward pass), the speedup per round is

$$
\text{speedup} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)\,(\gamma c + 1)}
$$

Both α and c enter, but two measurement studies find that c decides. [Ha et al. (ICML 2026)](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf) characterize five self-speculative methods on Llama-2 7B to 70B and state their first takeaway as "Draft cost, not acceptance rate, dominates throughput": the methods with the smallest drafters (Kangaroo, DEL) consistently beat the methods with better drafts (SWIFT, Draft & Verify). [Yan et al. (NAACL 2025)](https://arxiv.org/abs/2402.01528) reach the same conclusion for separate draft models: performance "depends heavily on the latency of the draft model", and the drafter's language-modeling quality "does not correlate strongly" with its speculative performance.

Existing drafters lower c in two ways. Self-drafters run fewer of the target's layers, by skipping layers ([Draft & Verify](https://aclanthology.org/2024.acl-long.607); SWIFT) or exiting early ([LayerSkip](https://aclanthology.org/2024.acl-long.681); [Kangaroo](https://arxiv.org/abs/2404.18911); DEL), and quantized self-drafters ([QSpec](https://arxiv.org/abs/2410.11305); [QuantSpec](https://arxiv.org/abs/2502.10424)) run the layers at 4 bits. Trained drafters ([EAGLE](https://arxiv.org/abs/2401.15077); EAGLE-3; DFlash) run a new small network fed with the target's features. DFlash reads features from five target layers and injects them into the keys and values of every draft layer, and reaches 6.1× on Qwen3-8B. The lesson of the trained drafters, stated in DFlash as "the target knows best", is that a drafter's quality comes mostly from the target's features and only a small share from the drafter's own arithmetic. Their arithmetic is still dense matrix products, and they need a new transformer trained on a large corpus.

MADDNESS removes the arithmetic. For a weight matrix known in advance it replaces the product with table lookups and 8-bit additions and performs zero multiply-adds; Blalock and Guttag report up to 100× over exact products on CPU. It was never adopted for exact inference because its error compounds through depth: [McCarter and Dronen (2022)](https://arxiv.org/abs/2207.05808) applied it to whole networks with fine-tuning and found accuracy "substantially diminished compared to exact matrix multiplication". A drafter is the one place where that failure is affordable. The target verifies every drafted token, so approximation error lowers α and leaves output quality unchanged, and the cost model above says α is the cheaper currency.

This proposal puts the target's features where a table drafter is weakest: the choice of prototype. Our TF's framing gives two levers for that step, better centroids or faster hashing, and the two trade off, because better centroids usually cost more to find and a faster hash usually loses accuracy. We push the first lever and hold the table budget fixed, so the result is an acceptance-rate gain at equal cost. The speed lever is a secondary exploration.

## Background and related work

Four lines of work meet here, and none of them has built a drafter from lookup tables. The table lists the methods this proposal is measured against and what each one changes about the drafter.

| Method | Drafter | How draft cost is cut | Extra training | Relation to this proposal |
| --- | --- | --- | --- | --- |
| [Draft & Verify](https://aclanthology.org/2024.acl-long.607) (ACL 2024) | Target with about half its layers skipped | Depth | None | Baseline. Draft cost 49% to 57% of the target per step (Ha et al.) |
| [LayerSkip](https://aclanthology.org/2024.acl-long.681) (ACL 2024) | Target's first 7 to 8 layers | Depth | Retrain the full model | Baseline. Draft cost 10% to 32% |
| [Kangaroo](https://arxiv.org/abs/2404.18911) (2024) | Target's first 2 to 6 layers plus a small adapter | Depth | Adapter only | The cost bar to beat: 8% to 13% |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) (NeurIPS 2025) | One trained layer fed features from three target layers | Separate small drafter | Train the drafter on a large corpus | Source of the feature-choice insight |
| [DFlash](https://arxiv.org/abs/2602.06036) (2026) | Five-layer block-diffusion drafter with five target layers injected into every layer's keys and values | One parallel pass per block | Train the drafter | Source of the fusion design we reuse; the speed bar |
| [QSpec](https://arxiv.org/abs/2410.11305) (EMNLP 2025) | Target at 4-bit weights and activations | Word size | None | Nearest quantized drafter, bits instead of tables |
| [QuantSpec](https://arxiv.org/abs/2502.10424) (ICML 2025) | Target at 4-bit weights with a 4-bit KV cache | Word size, KV bytes | None | Long-context baseline; acceptance above 90% |
| [MADDNESS](https://arxiv.org/abs/2106.10860), [LUT-NN](https://arxiv.org/abs/2302.03213), [Stella Nera](https://arxiv.org/abs/2311.10207) | Whole network through tables (not a drafter) | Multiplies become lookups | Centroid fine-tuning | The mechanism. Never applied to a decoder LLM or to drafting |

**Speculative decoding.** Leviathan et al. (2023) and Chen et al. (2023) introduced the draft-then-verify loop with a rejection rule that preserves the target's output distribution. [Medusa](https://arxiv.org/abs/2401.10774) and EAGLE (ICML 2024) replaced the separate drafter with heads trained on the target's own features; EAGLE's drafter takes the target's second-to-top-layer feature at the previous position plus the next token's embedding. EAGLE-3 fuses a low, a middle and a high target layer and reaches speedups up to 6.5×. DFlash fuses five target layers, spaced uniformly between the second layer and the third-to-last, through one projection W_c of shape D × 5D followed by RMSNorm, and injects the result into the key and value projections of every draft layer. These papers already ran the ablations on which target layers carry information about future tokens, and we reuse their choices rather than re-deriving them.

**Self-drafting.** Draft & Verify, LayerSkip, Kangaroo, SWIFT and DEL draft with a subset of the target's layers. Ha et al. (ICML 2026) measured all five on Llama-2 7B to 70B and found that the methods with the cheapest drafters win despite lower acceptance, and that early-exit methods also win on verification because the target reuses the draft layers' computation. Yan et al. (NAACL 2025) found the same for separate drafters.

**Lookup-based matrix products.** The course's framing of approximate matrix multiplication clusters whole activation rows into K centroids, finds the nearest centroid for each new row by K distance computations or by a locality-sensitive hash into K buckets, and looks up the precomputed product of that centroid with the weights. In that framing the online cost is step 1, the nearest-centroid search. MADDNESS is the product-quantized version, descended from [Jégou et al. (2011)](https://ieeexplore.ieee.org/document/5432202): it splits each row into C subvectors, assigns each to one of 16 prototypes with a four-level decision tree, and sums 8-bit table entries. The split moves the cost from step 1 to step 2, as the table below shows for one gate projection of Llama-3.2-1B (input width 2048, output width 8192).

| Step | Whole-row clustering | MADDNESS, subvector length 8 |
| --- | --- | --- |
| Exact product | 16.8M multiply-adds, 33.6 MB of 16-bit weights read | same |
| Step 1, assign to a centroid | K × 2048 operations for exact nearest centroid, or b × 2048 for a b-bit hash code | 1,024 scalar comparisons |
| Step 2, look up | 8,192 entries | 2.1M 8-bit reads and adds |

[LUT-NN](https://arxiv.org/abs/2302.03213) (MobiCom 2023) made the prototypes differentiable and ran whole networks on tables with up to 16× fewer FLOPs. Stella Nera built the accelerator (161 TOp/s/W) and fine-tuned through the lookups. McCarter and Dronen (2022) tried whole-network inference and found accuracy "substantially diminished". The LUT kernels now used for LLMs ([LUT-GEMM](https://arxiv.org/abs/2206.09557), ICLR 2024; [T-MAC](https://arxiv.org/abs/2407.00088); [FLUTE](https://aclanthology.org/2024.findings-emnlp.724), EMNLP 2024) index tables by weight bit patterns and build the tables from activations at run time, so they are exact for a quantized weight, and weight-side vector quantization ([AQLM](https://arxiv.org/abs/2401.06118), ICML 2024) puts the codebooks on the weights. MADDNESS indexes by activation codes, so its tables are fixed and precomputed and its error lives in the activations. We found no published application of activation-indexed tables to a decoder language model at or above 1B parameters.

**Quantized self-drafters.** QSpec drafts with 4-bit weights and activations and verifies with 4-bit weights at 16-bit activations, sharing weights and KV cache, for up to 1.64×. QuantSpec adds a hierarchical 4-bit KV cache for long contexts, reaching about 2.5×.

## Proposed method

The drafter is a smaller model from the target's family with every linear projection replaced by a lookup table. It shares the target's tokenizer, keeps its own KV cache, and receives a fused target feature after every verification. The outer loop is standard speculative decoding.

![One draft-verify round. The table drafter encodes, looks up and adds through every layer; the target verifies in one pass; accepted positions return tokens and a fused target feature that conditions the next round's encoder.](figures/draft_verify_round.png)

**Model pair.** Llama-3.2-1B drafts for Llama-3.1-8B. The two share the 128K-token vocabulary, which speculative decoding requires, and the same architecture family, which keeps the feature adapter small. Llama-3.2-3B for 8B and Qwen3-0.6B for Qwen3-8B are the secondary pairs.

**Table construction (offline, per projection).** A projection with weight W of shape d × m is split along its input dimension into C = d/L subvectors of length L. On about 100K activation rows collected from the drafter on 1M to 2M tokens of generic text, we fit K = 16 prototypes per subspace. Each table entry T_c[k, j] holds the dot product of prototype k in subspace c with the matching slice of column j of W, scaled to 8 bits as in MADDNESS. After the prototypes are fixed, the table entries are refit by ridge regression against the layer's exact outputs on the calibration set, which is closed-form and recovers part of the quantization error. Storage is 16·d·m/L bytes: equal to 8-bit weights at L = 16 and 4-bit weights at L = 32.

**Draft forward pass.** Per token and layer, the drafter encodes the input to C codes and, for each of the m outputs, sums C 8-bit table entries. The Q, K, V, O projections and the three MLP projections go through tables. Attention scores, softmax, RMSNorm, SiLU and residual additions stay exact in 16 bits; they are elementwise or scale with context length rather than with d·m. The LM head is tested both as a table and as an exact 8-bit product. After the drafter proposes γ tokens, the target runs one forward pass over them and applies the rejection rule of Leviathan et al., which keeps the output distribution exactly that of the target.

**Step 1: assigning activations to prototypes.** Three naive encoders are implemented as baselines. The Hamming encoder draws b = 4 random hyperplanes per subspace, hashes each subvector to a 4-bit code, and uses the mean of the calibration rows in that bucket as the centroid; this is the locality-sensitive hash of the lecture framing. The MADDNESS encoder is the four-level decision tree, four scalar comparisons per subvector. The nearest-prototype encoder computes all 16 distances, 16·L multiplies per subvector, and is the accuracy ceiling for a fixed codebook. All three cost under 1% of the exact layer's arithmetic. The difference between them is accuracy, and it is measured as acceptance rate.

**Feature-informed assignment.** After verification the target's features at every accepted position are free. Following DFlash, features from five target layers spaced uniformly through the depth are concatenated and projected by one matrix W_c of shape D_draft × 5 D_target with RMSNorm, giving a context feature f_t at the last accepted position. For Llama-3.1-8B into Llama-3.2-1B this is a 2048 × 20480 matrix, about 42M multiply-adds per accepted position, computed once per round for all accepted positions in parallel and reused across the γ draft steps. Two mechanisms use f_t, and both keep the bytes read per token unchanged.

- *Context-conditioned assignment.* A small per-layer adapter a_l maps f_t to a 32-dimensional context vector. The scorer for subspace c scores the 16 prototypes from the concatenation of the activation subvector and that context vector and takes the argmax. The scorer costs 16·(L + 32) multiply-adds per subvector, about 1% of the exact layer, and can later be distilled into a tree or a hash to restore the multiply-free property. Training is cross-entropy against the prototype whose table entry best reconstructs the exact layer output on the calibration set, with a second variant trained end to end on the drafter's next-token log-likelihood under the target.
- *Context-selected codebooks.* Each subspace holds M = 4 codebooks of 16 prototypes, fit by clustering the calibration rows after grouping them by a 2-bit hash of f_t. The context picks the codebook and the subvector picks the prototype within it. Storage grows by M; bytes read per token do not change. This is the "better centroids" lever in our TF's terms, and the first mechanism is the "better assignment" lever.

For draft positions beyond the first, f_t is either held at the last verified position (exact but stale by one token per step) or recomputed from the drafter's own approximate top-layer feature (fresh but noisy). Both are tested.

**Codebook budget per layer.** The subvector length L can differ by projection and layer. A projection that tolerates coarser assignment gets fewer codebooks. This is a tuning knob under a total-bytes budget, set from a one-off sensitivity sweep, and it is not a claimed contribution.

**What is not changed.** No target weight is modified, no new transformer layer is trained, and the output distribution is the target's. The trained components are the prototypes, the tables, the fusion projection, and the small per-layer adapters and scorers, a few tens of megabytes in total, trained on a calibration set rather than a corpus.

## Novelty

Three points, each stated against the closest prior work.

- **Feature-informed prototype assignment.** EAGLE, EAGLE-3 and DFlash feed the target's features into a trained transformer drafter. We feed them into the assignment step of a table drafter, so the signal that those methods spend a trained network on is used to pick one of 16 precomputed rows. As far as we can find, this is the first use of a verifier's exact features to steer a product-quantization encoder, and the first comparison of feature-informed assignment against naive locality-sensitive hashing on acceptance rate.
- **The first activation-indexed lookup-table drafter.** Every existing drafter cuts cost by depth (Draft & Verify, LayerSkip, Kangaroo, SWIFT, DEL), by word size (QSpec, QuantSpec), or by training a small dense network (EAGLE-3, DFlash). This drafter changes the arithmetic itself: its projections perform no multiply-adds and its error lives in the activations rather than the weights. The LUT kernels now used for LLMs (LUT-GEMM, T-MAC, FLUTE) are exact kernels for quantized weights and are a different object.
- **A setting where MADDNESS's known failure is affordable.** Whole-network lookup inference was tried and found to lose too much accuracy (McCarter and Dronen, 2022; Stella Nera recovers it only with fine-tuning on CIFAR-scale models). Speculative verification turns that accuracy loss into a lower acceptance rate with no change to outputs, and the measurements of Ha et al. and Yan et al. say acceptance rate is the cheaper currency. We have found no one who has made this argument or tested it.

A fourth point is a consequence rather than a claim: the tables and codes are the representation that LUT accelerators (Stella Nera) and CPU LUT kernels (T-MAC) already execute, and Gemma 4 ships a 76M to 500M parameter drafter with every model for on-device use, so a table drafter that works has a hardware path on phones that bit-quantized drafters do not.

## Feasibility and cost model

Decoding at batch size 1 is bound by memory bandwidth, so per-token cost is well approximated by bytes read, down to a floor set by kernel launch latency. A projection of shape d × m in 16-bit weights reads 2dm bytes per token. Its table holds C = d/L codebooks of 16 rows of m bytes, but a token reads only the row its code selects in each codebook: C·m = dm/L bytes, which is 1/(2L) of the 16-bit weights. The table is stored in full, so footprint and bytes read diverge, and the gap closes as batch size grows, because different tokens select different rows.

| Subvector length L | Table storage, as a fraction of 16-bit weights | Bytes read per token at batch 1, as a fraction of 16-bit weights |
| --- | --- | --- |
| 8 (MADDNESS default) | 1.0 | 1/16 |
| 16 | 0.5 | 1/32 |
| 32 | 0.25 | 1/64 |

For the primary pair the numbers are as follows. Llama-3.2-1B has about 0.97B parameters in linear layers (1.95 GB in 16 bits) plus a 263M-parameter LM head; Llama-3.1-8B reads about 16 GB per token in 16 bits.

| Drafter | Bytes read per drafted token | c by bytes | c expected on an H100 |
| --- | --- | --- | --- |
| Llama-3.2-1B, 16-bit | 2.5 GB | 0.16 | 0.15 to 0.3 |
| Llama-3.2-1B, 4-bit weights | 0.6 GB | 0.04 | 0.1 to 0.2 |
| Llama-3.2-1B, tables at L = 8, 8-bit LM head | 0.4 GB | 0.02 | 0.1 to 0.2 |

The last column is the honest one. A 16-layer forward pass at batch 1 is a few hundred kernel launches, and with CUDA graphs we estimate about 1 ms on an H100, against about 10 ms for the 8B target, so c has a floor near 0.1 that tables cannot lower. On a GPU the tables buy a modest latency gain over a 4-bit drafter and a large gain in bytes and energy; on bandwidth-bound hardware such as phones, where the lecture's MobileLLM and Apple Neural Engine material places the constraint, the byte reduction is the latency gain. This is why the main claim is the acceptance-rate gain at fixed budget, which holds on any hardware, and the speed result is measured and reported rather than promised.

**Batch size.** A dense weight matrix is read once per batch, but a table is read one row per codebook per token, and different tokens select different rows, so by a batch of 16 nearly the whole table has been read. Total bytes per projection per batch, as a fraction of the 16-bit weight matrix:

| Batch size | 16-bit | 4-bit | Tables, L = 8 | Tables, L = 16 | Tables, L = 32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.25 | 0.06 | 0.03 | 0.02 |
| 4 | 1 | 0.25 | 0.25 | 0.13 | 0.06 |
| 8 | 1 | 0.25 | 0.5 | 0.25 | 0.13 |
| 16 or more | 1 | 0.25 | 1 | 0.5 | 0.25 |

The byte advantage over a 4-bit drafter therefore holds up to batch 4 at L = 8 and batch 16 at L = 32, which covers interactive serving and on-device decoding. At larger batches the table drafter is bound by its 8-bit additions, which are L times fewer operations than the dense product but do not run on tensor cores, so a quantized tensor-core drafter is expected to be faster there; this is measured, not claimed. Speculative decoding as a whole also weakens at large batch, because a compute-bound target pays close to γ forward passes to verify γ tokens, which raises c for every drafter. The acceptance-rate claim does not depend on batch size.

The speedup formula from the Motivation gives the targets. With γ = 5 drafted tokens per round:

| Drafter cost c | α = 0.6 | α = 0.7 | α = 0.8 |
| --- | --- | --- | --- |
| 0.30 | 0.95× | 1.18× | 1.48× |
| 0.15 | 1.36× | 1.68× | 2.11× |
| 0.10 | 1.59× | 1.96× | 2.46× |
| 0.05 | 1.91× | 2.35× | 2.95× |

We expect a 16-bit Llama-3.2-1B drafter to reach α of roughly 0.6 to 0.7 under greedy verification for the 8B target; experiment 2 measures it, and it is the ceiling for the table drafter at any budget. The table drafter is worthwhile if naive assignment lands within 0.1 of that ceiling at L = 8, and feature-informed assignment then closes the gap or reaches the same α at L = 16.

Four things can break the model, and the first experiments are designed around them.

- **Encoder latency.** The tree encoder costs four comparisons per subvector, which is nothing in arithmetic, but as a separate kernel it is a chain of dependent steps per projection: 7 projections × 16 layers × 4 levels is about 450 dependent launches per drafted token. The encoder must be fused into the gather-and-add kernel of each projection. The scorer of the feature-informed encoder is a single small dense product per projection and fuses more easily.
- **Activation outliers.** LLM activations have a few channels with values 10× to 100× the rest, which is why activation quantization is harder than weight quantization. K-means on raw subvectors would spend its 16 prototypes on those channels. Per-channel scaling of the kind used by [SmoothQuant](https://arxiv.org/abs/2211.10438) folds into the weight before the table is built, so it is free here; whether it is enough is measured in the first experiment.
- **The fusion projection.** W_c is a dense 2048 × 20480 product. Computed once per round over the accepted positions it is one small matrix product per verification, and it must stay out of the per-draft-token path. If it is needed per draft step it becomes a table like the others.
- **No reuse at verification.** A separate drafter keeps its own KV cache and the target runs all its layers from scratch, unlike early-exit self-drafters. The drafter earns its place on draft cost and acceptance alone.

## Further explorations

The first group extends the method; the second uses the artifact to answer questions about the models.

**Extensions**

- **Faster hashing.** Our TF's second lever. Once the accuracy result is in, the scorer can be distilled into a tree or a Hamming code at matched acceptance, which restores the multiply-free encoder and measures how much accuracy the context buys per comparison.
- **Which target features.** EAGLE uses the second-to-top layer, EAGLE-3 a low, middle and high layer, DFlash five uniformly spaced layers. The same ablation on the assignment task tells whether a hash wants the same features a trained drafter wants.
- **Table fine-tuning by distillation.** LUT-NN and Stella Nera make the lookups differentiable. Here the tables, prototypes and scorers are the only trained parameters, so fine-tuning them to match the target's next-token distribution on a few million tokens is cheap and attacks α directly.
- **Learned subspaces and rotations.** MADDNESS groups adjacent input dimensions. Choosing the grouping, or applying an orthogonal rotation of the input as [QuaRot](https://arxiv.org/abs/2404.00456) does to remove outliers, folds into the weight before the table is built and costs nothing at run time.
- **Tree drafting.** With c near 0.1 the drafter can afford to propose a tree of candidates as EAGLE-2 does instead of a chain. Tree width raises the number of table rows read per round, so the byte advantage fades around 16 candidates.
- **Batched serving.** Past the batch sizes in the Feasibility table the drafter is bound by 8-bit additions rather than bytes. Whether it still beats a 4-bit drafter there decides whether the method is a latency tool or also a throughput tool.
- **Hardware.** The tables and codes are what Stella Nera and T-MAC execute. A CPU-side or neural-engine drafter on a phone, paired with a GPU or server target, is where the multiply-free property matters most.

**Questions the artifact can answer**

- **A per-projection codebook map.** Acceptance rate as a function of L for each projection and layer measures how many distinct input patterns each projection needs for next-token prediction. This was the layer-compression idea in the first draft of this proposal; it is now a by-product rather than a contribution.
- **Temporal predictability of activations.** The scorer's accuracy at predicting layer-l codes from the previous position's fused target feature measures, per layer, how much of the current activation is already determined by context.
- **Error compounding through depth.** Comparing a drafter with tables in all layers to one with tables in half of them, at matched bytes, measures how approximation error accumulates in a decoder, which is the quantity McCarter and Dronen found prohibitive for exact inference.

## Starter experiments

The first three experiments run on one 80 GB GPU and decide whether the project continues; the rest build the paper. Calibration uses 1M to 2M tokens of FineWeb or WikiText; evaluation uses CNN/DailyMail, GSM8K and HumanEval prompts, the three datasets of Ha et al. Every result table reports c, α and tokens per second together, as Ha et al. recommend.

1. **Single-layer reconstruction.** For one middle-layer MLP down-projection and one attention output projection of Llama-3.2-1B, fit MADDNESS at L of 8, 16 and 32, with and without SmoothQuant-style per-channel scaling, and refit the tables by ridge regression. Measure relative output error and the KL divergence of the drafter's next-token distribution when only that layer is replaced. Compare with 8-bit and 4-bit weight quantization of the same layer. Pure PyTorch gather-and-sum, no kernels. Gate: at L = 8 the KL cost is within 2× that of 4-bit weights; otherwise outlier handling comes first.
2. **Acceptance rate under naive assignment.** Replace every projection at uniform L and measure α with γ = 5 under greedy and sampled verification against Llama-3.1-8B, for the Hamming, tree and nearest-prototype encoders. Also measure the 16-bit and 4-bit Llama-3.2-1B drafters, which set the ceiling. Gate: the tree encoder at L = 8 lands within 0.1 of the 16-bit ceiling. Below that, table fine-tuning (experiment 7) moves ahead of everything else.
3. **Feature-informed assignment.** Train the fusion projection, per-layer adapters and per-subspace scorers on the calibration set against teacher codes, then the context-selected codebook variant. Measure code accuracy and α at L of 8, 16 and 32 against the naive encoders at the same L, with f_t held at the last verified position and with it recomputed from the drafter's own feature. This is the main result: the α gain at equal L, or the L at which the conditioned encoder matches the naive encoder's α. Gate: a gain of 0.05 in α at L = 8, or matching α at L = 16.
4. **Fused kernel microbenchmark (in parallel with 2 and 3).** Write one Triton kernel per projection that encodes, gathers the selected rows and adds them in 8 bits, at batch size 1 on shapes 2048 × 2048 and 2048 × 8192. Time it against a BF16 matrix-vector product and a Marlin 4-bit kernel; report bytes read and achieved bandwidth. Gate: per-projection time at or below the 4-bit kernel at L = 8.
5. **End-to-end speedup.** Run drafter and target in one loop with CUDA graphs. Report c, α and tokens per second against vanilla decoding, the 16-bit and 4-bit Llama-3.2-1B drafters, and Draft & Verify and LayerSkip from their released code. EAGLE-3 and DFlash numbers on Llama-3.1-8B are quoted from their papers as the trained-drafter bar.
6. **Which target features.** Repeat experiment 3 with EAGLE's second-to-top layer, EAGLE-3's three layers, and DFlash's five, at matched adapter size.
7. **Table fine-tuning by distillation (if needed).** Fine-tune tables, prototypes and scorers through a straight-through estimator on 5M tokens against the target's next-token distribution and report the gain in α.

## Open questions

Questions about the design that the experiments or the TF will settle.

- **The adapter's input.** Which target layers, and only the last accepted position or every position in the accepted block? DFlash's five uniform layers are the starting point; experiment 6 is the ablation.
- **The adapter's output.** The context vector can enter the scorer beside the activation subvector, pick the codebook, or be injected into the drafter's keys and values as DFlash does. The first two are in scope; the third makes the drafter a trained model and is out of scope.
- **The adapter's training.** Drafter and tables frozen with only the adapters and scorers trained, and which loss: cross-entropy against the exact encoder's codes, or KL to the target's next-token distribution. If code accuracy improves and α does not, the scorer is predicting codes that do not matter and the loss switches to the second.
- **The adapter's cost.** W_c at 2048 × 20480 is as large as one target MLP projection per accepted position. It must run once per round, not per draft step; whether the stale f_t this implies costs acceptance is measured in experiment 3.
- **Why same family.** Whether the drafter must share the target's family for dimension and tokenizer matching only, or because the two models' representations align and keep the adapter small. Qwen3-0.6B for Qwen3-8B tests the second within a family; a cross-family pair with a shared tokenizer would test it across, and none is planned.
- **Does the Hamming baseline need several hash tables?** A single 4-bit random-hyperplane hash per subspace may assign far from the nearest centroid in an 8-dimensional subspace. If so, the baseline is given the number of tables that matches the tree encoder's accuracy, so the comparison is at equal naive accuracy.
- **Can the encoder be fused on a GPU?** A tree with data-dependent branches inside a kernel causes warp divergence. The scorer is branch-free and may be the faster encoder on a GPU despite its multiplies. The multiply-free property matters for hardware; the GPU result does not depend on it.
- **Sampled verification.** Under sampling, acceptance depends on the drafter's full distribution, and a table drafter's logits are distorted in ways that greedy acceptance hides. Temperature 0 and 1 are both measured.
- **Which target precision is the baseline?** Ha et al. use 16-bit targets. Against a target served at 4-bit weights every drafter's c rises by about 4×, and the right comparison for a paper is both.

Logistics to settle with the TF: which GPUs and how many hours are available; whether a PyTorch simulation of the drafter plus a kernel microbenchmark is an acceptable deliverable if the end-to-end loop slips; whether he knows a usable GPU implementation of MADDNESS beyond the Stella Nera group's halutmatmul; and whether he knows any work applying activation-side product quantization to transformers that we should cite or avoid duplicating.

## References

Every entry was checked against its arXiv or venue page on 2026-10-26 or 2026-10-27. Where a venue could not be confirmed the arXiv listing is given.

- Ashkboos, S. et al. (2024). QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Blalock, D. and Guttag, J. (2021). Multiplying Matrices Without Multiplying. ICML 2021, PMLR 139. [arXiv:2106.10860](https://arxiv.org/abs/2106.10860)
- Cai, T. et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L. and Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Chen, J., Liang, Y. and Liu, Z. (2026). DFlash: Block Diffusion for Flash Speculative Decoding. [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)
- Egiazarian, V. et al. (2024). Extreme Compression of Large Language Models via Additive Quantization. ICML 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118)
- Elhoushi, M. et al. (2024). LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.681)
- Guo, H. et al. (2024). Fast Matrix Multiplications for Lookup Table-Quantized LLMs. Findings of EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.724)
- Ha, J., Ganesan, K., Nguyen, A., Ahmed, T. and Moshovos, A. (2026). Characterizing Self-Speculative Decoding Approaches for Accelerating LLMs. ICML 2026. [PDF](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf)
- Jégou, H., Douze, M. and Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. IEEE TPAMI 33(1). [IEEE](https://ieeexplore.ieee.org/document/5432202)
- Kung, H. T. (2026). CS 2420 lecture notes: approximate and multiplication-free matrix multiplication; lecture 14, MobileLLM, Gemma 4, Diffusion LLMs (26 October 2026). Harvard University.
- Leviathan, Y., Kalman, M. and Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023, PMLR 202. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Li, Y., Wei, F., Zhang, C. and Zhang, H. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. ICML 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Li, Y., Wei, F., Zhang, C. and Zhang, H. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test. NeurIPS 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- Liu, F. et al. (2024). Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting. NeurIPS 2024. [arXiv:2404.18911](https://arxiv.org/abs/2404.18911)
- McCarter, C. and Dronen, N. (2022). Look-ups are not (yet) all you need for deep learning inference. [arXiv:2207.05808](https://arxiv.org/abs/2207.05808)
- Park, G. et al. (2024). LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models. ICLR 2024. [arXiv:2206.09557](https://arxiv.org/abs/2206.09557)
- Schönleber, J. et al. (2023). Stella Nera: A Differentiable Maddness-Based Hardware Accelerator for Efficient Approximate Matrix Multiplication. ISVLSI 2025. [arXiv:2311.10207](https://arxiv.org/abs/2311.10207)
- Tang, X. et al. (2023). LUT-NN: Empower Efficient Neural Network Inference with Centroid Learning and Table Lookup. MobiCom 2023. [arXiv:2302.03213](https://arxiv.org/abs/2302.03213)
- Tiwari, R. et al. (2025). QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache. ICML 2025, PMLR 267. [arXiv:2502.10424](https://arxiv.org/abs/2502.10424)
- Wei, J. et al. (2024). T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge. [arXiv:2407.00088](https://arxiv.org/abs/2407.00088)
- Xiao, G. et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- Yan, M., Agarwal, S. and Venkataraman, S. (2025). Decoding Speculative Decoding. NAACL 2025. [arXiv:2402.01528](https://arxiv.org/abs/2402.01528)
- Zhang, J. et al. (2024). Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding. ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.607)
- Zhao, J., Lu, W., Wang, S., Kong, L. and Wu, C. (2025). QSpec: Speculative Decoding with Complementary Quantization Schemes. EMNLP 2025. [arXiv:2410.11305](https://arxiv.org/abs/2410.11305)

Named in the text without their own entry, as characterized by Ha et al. (2026): SWIFT (Xia et al., 2025) and DEL (Zarch et al., 2025). EAGLE-2 is named for its tree drafting and is described in the EAGLE-3 paper.
