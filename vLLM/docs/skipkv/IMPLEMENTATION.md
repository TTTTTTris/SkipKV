# SkipKV sentence eviction & eviction granularity — design (vLLM port)

The design decisions are summarized in [`skipkv/DESIGN.md`](./DESIGN.md). This document
covers what was added on top of the R-KV port ([`rkv/DESIGN.md`](../rkv/DESIGN.md),
[`rkv/IMPLEMENTATION.md`](../rkv/IMPLEMENTATION.md)) in detail:

1. **SkipKV sentence eviction** ([MLSys 2026: SkipKV](https://openreview.net/pdf?id=0EsV9SIm8p)) — evict
   whole redundant sentences, on top of R-KV's token score (§1–§5).
2. **Keep-first variant** used as an ablation (§6).
3. **Eviction granularity** — one global kept set (default) or a kept set per (layer, KV head)
   (§7). It applies to plain R-KV as well as SkipKV.
4. **Prompt-retention diagnostics** (§8).
5. **SkipKV activation steering** — the other half of SkipKV (§13). It also runs without R-KV
   (Full-KV + steering).

Code: [`skipkv/sentence.py`](../../skipkv/sentence.py) (tracker, no vLLM deps),
[`skipkv/steering.py`](../../skipkv/steering.py) (steering, no vLLM deps),
[`skipkv/integration.py`](../../skipkv/integration.py) (`RKVConfig`, `RKVCompressor.compact_step`),
runner wiring in `vllm/v1/worker/gpu_model_runner.py` (via the patch).

---

## 1. The method

Reasoning models repeat themselves: the same derivation step or re-statement appears several
times. R-KV scores tokens one at a time, so a repeated sentence loses a few tokens here and there
and leaves fragments behind. SkipKV adds a sentence-level signal:

- **Segmentation.** A sentence ends at a boundary token: the last token of `"\n"`, `".\n"`,
  `")\n"`, `"\n\n"`, `".\n\n"`, `")\n\n"` under the model's tokenizer (`boundary_token_ids`). A
  boundary token right after another boundary (e.g. `".\n\n"` split in two) extends the previous
  sentence instead of opening an empty one.
- **Representation.** The mean of the sentence's **final-layer hidden states**, L2-normalized.
- **Redundancy.** When a sentence closes it is compared (cosine) against every earlier sentence of
  the same request. For each pair above `threshold` (0.95 = the paper's value), the **earlier**
  sentence is marked redundant with penalty = similarity, so the latest occurrence survives. This
  matches the official implementation (`cal_sentence_similarity_from_reprs`, "mark EARLIER ranges").
- **Eviction.** At compaction, every token of a redundant sentence has its R-KV score reduced by
  the penalty. R-KV scores are softmax-scale (≈1/seq_len), so a 0.95 penalty always wins and the
  sentence is evicted as a unit; non-redundant tokens are ranked by R-KV as before.

## 2. Where it runs

```
GPUModelRunner.execute_model
  ├─ self.model(...)                               # full forward, final hidden states
  ├─ SentenceTracker.update(req_ids, query_start_loc,
  │                         num_computed_tokens,   # logical position of each row's first token
  │                         input_ids, hidden)     # segment + represent + compare
  └─ RKVCompressor.compact_step(..., req_ids, logical_ends, prompt_lens)
       └─ SentenceTracker.penalties(req_id, logical positions of the cached entries)
```

The tracker is created by `RKVCompressor.enable_sentences(boundary_ids)`, which the runner calls
at start-up when `VLLM_V1_SKIPKV_SENTENCE_THRESHOLD > 0` (it loads the tokenizer to find the
boundary ids). With the threshold at 0 nothing is created and R-KV is unchanged.

SkipKV's additions to the shared wiring patch (the R-KV wiring is in
[`rkv/IMPLEMENTATION.md`](../rkv/IMPLEMENTATION.md) §3):

| File | SkipKV change |
| --- | --- |
| `vllm/v1/worker/gpu_model_runner.py` | Sentence eviction: build the tracker at start-up (tokenizer boundary ids), feed it each step's token ids + final hidden states after the forward, and pass `req_ids`, per-request logical end positions and prompt lengths to `compact_step`. Steering: `_rkv_setup_steering` installs the vector, the per-token coefficient buffer and the steered layer on the model; each step plans the coefficients from the token ids and releases finished requests. |
| `vllm/model_executor/models/qwen2.py` | Steering hook: `steer_residual` on the input of the steered decoder layer (or after the last layer). Qwen2 architecture only. |
| `vllm/envs.py` | `VLLM_V1_SKIPKV_STEER_VECTOR` and `VLLM_V1_SKIPKV_STEER_LAYER` (declared so they enter the `torch.compile` cache key: they change the graph); the other SkipKV variables are read by `RKVConfig.from_env` / `SteeringConfig.from_env`. |
| `vllm/config/vllm.py` | Steering selects the V1 runner and disables async scheduling, also without R-KV (Full-KV + steering). |

## 3. Logical positions across compaction

Sentences are identified by **logical** token positions (the RoPE positions the runner assigns),
because physical slots are reused after every compaction. The compressor keeps, per request,
`_logpos[req_id]`: the logical position of each kept entry after the last compaction.
`_logical_positions(req_id, seq_len, logical_end)` rebuilds the full map for the current step as
`[kept entries] + [contiguous new tokens up to logical_end]`, and `compact_step` stores the
positions of the newly kept set. Penalties are looked up by those positions, so a sentence keeps
its identity however many times it has been compacted.

`_logpos` is only maintained when something needs it (a sentence tracker, or the §8 diagnostics),
and is dropped on `release_requests` (finished) / `reset_request_windows` (preempted —
recomputation replays the sequence and the tracker re-segments it).

## 4. Tracker performance design

The tracker runs on **every** model step, so it avoids GPU→CPU syncs and per-request Python:

- **Decode steps never block.** Running sums, sentence means and resets are done on the GPU. The
  boundary mask is copied to pinned host memory asynchronously; its host bookkeeping (sentence
  start/end, lone-boundary merges) is applied at the **start of the next step**, when the copy has
  long finished. Consequence: compaction sees sentences closed up to the previous step (a one-step
  lag, at most one sentence per request).
- **No per-request Python.** Host state lives in numpy arrays indexed by a per-request slot; only
  the rare lone-boundary merge loops.
- **One batched comparison.** Closed sentences live in a shared pool (normalized representation,
  owner slot, allocation order, penalty). All sentences closed in a step are compared with one
  `(closed × pool)` matmul, masked to "same request, closed earlier", and penalties are max-updated
  on the GPU with no readback.
- **Prefill chunks** (multi-token rows) take a synchronous path; they only occur during admission.
- **Only cached sentences are compared.** Each pair is compared once, when the later sentence closes
  (compaction only looks up stored penalties). After every compaction, a sentence with no token
  left in the cache — in any layer or head — is dropped from the pool (`evicted_rows` /
  `free_evicted`, one host sync per compaction step), so new sentences are compared only with
  sentences still cached, and the pool is bounded by what is cached instead of by output length.
  The request's most recent sentence is never dropped (a following lone boundary token extends it).
  This is exact for keep-last (a penalty on an evicted sentence has no effect);
  with keep-first it changes results (a repeat of an evicted sentence is no longer penalized).
  Disable with `VLLM_V1_SKIPKV_SENTENCE_PRUNE=0`. The HuggingFace SkipKV code keeps every sentence
  representation until the request ends (it only drops evicted sentences from each head's
  position map), so this is a memory optimization over the reference, not a behaviour change.

Measured on R1-Distill-Qwen-7B GSM8K (budget 256 / buffer 128, 3 interleaved repeats): SkipKV
costs **~3%** of R-KV's decode throughput (it was ~7% with the first, per-step-syncing tracker).

**Invariant (bug found in review):** `penalties()` must flush the deferred step
**before** reading the pool arrays — the flush can commit sentences that grow (reallocate) them,
and indexing the old array with the new row ids is out of bounds. `_per_position` takes the array
by attribute name and reads it after `_flush()`; the flush is unconditional (compaction already
syncs), which also keeps results independent of GPU timing.

## 5. Penalty scale

`compact_step` subtracts `scale * penalty`. In **global** granularity the score is a sum over
layers of the head-mean score, so `scale = num_layers` reproduces the reference, which subtracts
the penalty from every layer's per-head score. In **per-head** granularity (§7) each score is one
(layer, head)'s, so `scale = 1`.

## 6. Variant (ablation)

| Variant | Env | What changes |
| --- | --- | --- |
| Keep-first | `VLLM_V1_SKIPKV_SENTENCE_KEEP=first` | Penalize the **later** sentence of a similar pair instead of the earlier one (default `last` = the paper). Protects the prompt, which the reasoning tends to paraphrase early. |

Finding (R1-Distill-Qwen-7B): keep-first does not help SkipKV.

## 7. Eviction granularity: global vs. per-(layer, KV head)

`VLLM_V1_SKIPKV_GRANULARITY=global|head` (default `global`).

- **global** (the SGLang port's choice): score per layer, **mean over KV heads, sum over layers**,
  one kept set; every layer and head relocates the same tokens. (Note: the HuggingFace R-KV
  reference does *not* do this — `update_kv` keeps a top-k per layer and per KV head.)
- **head** (the HuggingFace reference): every (layer, KV head) keeps its own top
  `budget − window` tokens plus the window.

Why per-head fits vLLM's paged cache: every (layer, head) still keeps exactly `budget` entries,
so `seq_len`, the block table and `num_dropped_tokens` stay shared per request. Only *which token
sits in a slot* differs per (layer, head), and attention does not care — keys are stored with RoPE
already applied, a decode query attends to every cached slot in any order, and each head has its
own softmax. New tokens get their position from the logical position, not the slot.

Implementation:

- `_score_group(..., per_head=True)` returns scores `(num_reqs, num_layers, kv_heads, P)`
  unreduced; selection is a top-k along the last axis.
- `_relocate_per_head` gathers/scatters per (layer, head): `key_cache[blk, off, head]`, all heads
  gathered before any scatter (alias-safe), same `budget` destination slots for every head.
- `_logpos[req_id]` becomes `(num_layers, kv_heads, budget)`; `_logical_positions` appends new
  tokens per head, and penalty lookups index it with `[..., :P]` (slicing the first axis instead
  crashed the first per-head SkipKV runs — fixed, and covered by a test).
- Tensor parallelism: per-head scores are local to each rank's heads, so the score all-reduce is
  skipped.
- Not supported with `score_mode=reference` (config error).

Findings (R1-Distill-Qwen-7B): per-head wins big **when the budget is small relative to the
prompt**, and is neutral otherwise, at the same speed.

| Task | Budgets | Per-head − global |
| --- | --- | --- |
| GSM8K (prompt 80–245 tokens) | 128, 256 | +5 to +34 points |
| MATH-500 | 256 / 512 / 1024 / 2048 | +12 / +2–6 / ≈0 / ≈0 |

The §8 diagnostics show why: at MATH-500 budget 256 the global kept set holds 25% of the prompt,
while per-head heads each keep ~20% but **together** keep 98% (14× the budget in distinct tokens).
The gain disappears once the global set already keeps most of the prompt (≥80% from budget 1024).
This is correlational; the direct test (global with the prompt never evicted) has not been run.

## 8. Prompt-retention diagnostics

`VLLM_V1_SKIPKV_PROMPT_STATS=1` records, at every compaction, from the kept logical positions:
the mean fraction of the prompt kept per (layer, head), the fraction kept by **at least one**
head, and distinct kept tokens ÷ budget (1.0 in global mode; a measure of head disagreement).
Accumulated on the device (no sync); read with `RKVCompressor.prompt_stats`. Needs each request's
prompt length, which the runner passes as `compact_step(..., prompt_lens=...)`.

## 9. Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `VLLM_V1_SKIPKV_SENTENCE_THRESHOLD` | `0` (off) | Enables SkipKV; cosine threshold for a redundant pair (paper: 0.95). |
| `VLLM_V1_SKIPKV_SENTENCE_KEEP` | `last` | `last` = evict the earlier copy (paper); `first` = evict the later copy. |
| `VLLM_V1_SKIPKV_SENTENCE_PRUNE` | `1` | Drop sentences with no cached token from the pool (compare new sentences only with cached ones). |
| `VLLM_V1_SKIPKV_GRANULARITY` | `global` | `head` = per-(layer, KV head) kept sets (R-KV and SkipKV). |
| `VLLM_V1_SKIPKV_PROMPT_STATS` | `0` | `1` = record prompt-retention diagnostics. |

These are read by `RKVConfig.from_env` directly, not registered in `vllm/envs.py`, so vLLM logs
"Unknown vLLM environment variable" for them; that warning is harmless.

**Settings that silently invalidate long-output results** (both hit during long-output runs):

- `VLLM_V1_R_KV_FREE_BLOCKS=1` for long outputs. Without it evicted blocks are never returned, the
  cache fills, requests are preempted and recomputed at full length, and R-KV/SkipKV then leave
  them uncompressed (over the scoring-memory cap, "never compacted") — 15–32% of compactions went
  missing and accuracy looked inflated. Check: peak KV usage ≈ (budget + buffer) ×
  running requests, and logged compactions ≈ the count simulated from output lengths.
- `VLLM_V1_R_KV_SCORE_CHUNK_MB=2048` for budgets ≥ ~2.5K; the default 512 MB cap silently leaves
  never-compacted requests at Full-KV.

## 10. Results: accuracy vs throughput

DeepSeek-R1-Distill-Qwen-7B, one H100 NVL, all compressed runs with evicted blocks freed. Each cell
is **accuracy / wall-time speedup vs Full-KV** (Full-KV wall time ÷ run wall time for the whole
batch). Full-KV has prefix caching off, as R-KV forces it off.

**GSM8K, buffer 128** (Full-KV 89.2% / 1.00×, 54 s)

| Method | budget 128 | budget 256 | budget 512 |
| --- | --- | --- | --- |
| R-KV global | 43.6% / 0.50× | 80.8% / 0.53× | 89.2% / 0.53× |
| R-KV per-head | 57.4% / 0.51× | 85.7% / 0.56× | — |
| SkipKV global | 44.5% / 0.48× | 82.1% / 0.50× | 89.2% / 0.50× |
| SkipKV per-head | 61.3% / 0.50× | 86.0% / 0.51× | — |

**GSM8K, buffer 16** (Full-KV 89.2% / 1.00×, 54 s)

| Method | budget 128 | budget 256 | budget 512 |
| --- | --- | --- | --- |
| R-KV global | 23.4% / 0.38× | 75.7% / 0.44× | 89.2% / 0.48× |
| R-KV per-head | 57.2% / 0.40× | 86.5% / 0.47× | — |
| SkipKV global | 22.3% / 0.36× | 79.3% / 0.41× | 88.9% / 0.44× |
| SkipKV per-head | 56.1% / 0.40× | 86.3% / 0.41× | — |

**MATH-500, buffer 128, greedy** (Full-KV 87.2% / 1.00×, 302 s)

| Method | budget 256 | budget 512 | budget 1024 | budget 2048 |
| --- | --- | --- | --- | --- |
| R-KV global | 44.8% / 1.02× | 66.8% / 1.09× | 79.6% / 1.11× | 84.2% / 0.93× |
| R-KV per-head | 56.6% / 1.08× | 68.8% / 1.12× | 78.4% / 1.11× | 85.4% / 0.94× |
| SkipKV global | 47.8% / 1.00× | 66.8% / 1.08× | 79.8% / 1.08× | 86.0% / 0.98× |
| SkipKV per-head | 60.2% / 1.08× | 72.8% / 1.12× | 81.2% / 1.06× | 86.4% / 0.96× |

Per-head was not run at GSM8K budget 512.

Takeaways:

- **Speedup needs memory pressure.** On this H100, GSM8K and MATH-500 fit in Full-KV's cache, so
  compression cannot speed them up: GSM8K runs are 0.2–0.8× Full-KV (compression only adds work);
  MATH-500 stays near 1× because tighter budgets make answers longer (6–8K vs 3.2K tokens).
- **Per-head beats global when the budget is small relative to the prompt** (GSM8K 128–256, MATH-500
  256–512), at the same speed; elsewhere they tie (§7).
- **SkipKV vs R-KV** (same granularity): SkipKV wins most clearly under global eviction at tight
  budgets (GSM8K budget 256: +1.3 to +3.6 points); per-head, the two are close, with SkipKV ahead on
  MATH-500 at budgets 256–1024. SkipKV costs ~3% of decode speed (§4).

Caveats: one run per point. Throughput was measured on a shared machine (runs paired on two GPUs;
GSM8K methods timed on different days), so speed gaps under ~10% are noise. MATH-500 accuracy has
about ±2 points of run-to-run noise.

## 11. Validation

- CPU tests: [`tests/test_skipkv.py`](../../tests/test_skipkv.py) (segmentation, chunk/decode
  equivalence, mixed prefill+decode steps, lone-boundary merges, pool growth, release, keep-first,
  penalized eviction, logical positions across compactions),
  [`tests/test_rkv_per_head.py`](../../tests/test_rkv_per_head.py) (per-(layer, head) kept sets equal
  the HuggingFace per-head top-k; global keeps one set; SkipKV positions and penalty lookup across
  per-head compactions; prompt stats checked against hand-computed values).
- GPU: the deferred tracker matched the synchronous one exactly (40 staggered requests, same
  sentences, penalties and redundant set); the pool-growth crash was reproduced and fixed on GPU.
- Wiring check: with threshold 1.0 no sentence can be marked redundant, so SkipKV must behave
  exactly like R-KV. On R1-Distill-Qwen-7B, full GSM8K (1319 questions), budget 256, it produced
  byte-identical text on every question with the same compaction count: the tracker, position
  bookkeeping and penalty plumbing are transparent when inactive.
- Every reported run was checked: logged compactions within 0.2–2% of the count simulated from
  output lengths, and peak KV usage within budget.

## 12. Known limitations

- Activation steering is wired only into Qwen2-architecture models (`qwen2.py`; e.g.
  R1-Distill-Qwen). Other architectures fail at start-up with a clear error.
- `InputBatch.swap_states` does not move `num_dropped_tokens_cpu` / `should_compress_list`. Harmless
  with FlashAttention (it never reorders the batch); would misalign them for backends that do.
- vLLM's own "Preemptions" stat reported 0 even when preemption happened: an upstream v0.25.1
  bug (`LoggingStatLogger.log()` reset the count before printing it). Fixed in the patch;
  `eval.py --stats` now reports the run's total preemptions (checked against a direct count
  of `Scheduler._preempt_request` calls: 35 = 35).
- Per-head relocation moves the same bytes but indexes 4× more entries (one set per KV head); a
  small cost may exist at very frequent compaction (not confirmed with repeats).

## 13. Activation steering

SkipKV also steers generation away from redundant reflection, SEAL-style
([`skipkv/steering.py`](../../skipkv/steering.py)), following the reference `skipkv/modeling.py`:

- **What is added.** A precomputed vector `v` (SEAL's "transition/reflection" direction; for
  R1-Distill-Qwen-7B `SkipKV/steering_vectors/DeepSeek-R1-Distill-Qwen-7B/layer_20_transition_reflection_steervec.pt`)
  is added to the residual stream entering decoder layer `L`: `h += coef * v`.
- **Where.** Only at paragraph-break tokens (vocab tokens whose text contains `"\n\n"`), only inside
  `<think> … </think>`; in the prompt, only its last token (as the reference, which steers the last
  position of each forward).
- **How strongly.** `coef = coef_init − gamma × n_check`, where `n_check` counts sentences containing
  a check word (`Alternatively`, `Wait`, `again`); sentences end at `"\n"`, `".\n"`, `")\n"`,
  `"\n\n"`, `".\n\n"`, `")\n\n"`, `"?\n\n"`. `coef` is refreshed on the reference's compression
  steps — after the token at position `p` when `(p + 1) % 128 == 0` (`divide_method="step_length"`,
  `divide_length=128`, its default) — and used from the next token on. Defaults are the reference's
  R1-Qwen-7B settings: `coef_init = −1.25`, `gamma = 0.02`, `L = 20`.
- **Per request.** The reference keeps one coefficient and one length counter per batch; here both are
  per request, which equals the reference at batch size 1.

**Implementation.** `Qwen2Model.forward` (patched) adds `coef[:, None] * v` to the residual before
layer `L`; vLLM layers carry the stream as `hidden_states + residual`, so this equals the reference's
`hidden_states += coef * v`. `coef` is a persistent per-token GPU buffer (0 = not steered) and `v` a
model buffer, so CUDA graphs see each step's values. The host-side `SteeringController` fills the
buffer once per step from the token ids the runner already holds (no GPU sync). Its per-request state
is a function of the token prefix, so recomputed requests (preemption, prefix-cache hits) are
re-scanned and their replayed positions steered exactly as the first time.

Steering forces the **V1 model runner** and **synchronous scheduling** (it reads each step's token ids
on the host). `VLLM_V1_SKIPKV_STEER_VECTOR` and `_LAYER` are registered in `vllm/envs.py` so they enter
the torch.compile cache key: without that, a graph compiled with steering off was reused from the
cache and steering silently did nothing (caught by the GPU check below).

| Env | Default | Meaning |
| --- | --- | --- |
| `VLLM_V1_SKIPKV_STEER_VECTOR` | unset (off) | Path of the steering vector (`.pt`, hidden-size elements). Enables steering. |
| `VLLM_V1_SKIPKV_STEER_LAYER` | `20` | Decoder layer whose input is steered (`num_layers` = after the last layer). |
| `VLLM_V1_SKIPKV_STEER_COEF` | `-1.25` | `coef_init`. |
| `VLLM_V1_SKIPKV_STEER_GAMMA` | `0.02` | Coefficient change per check sentence. |
| `VLLM_V1_SKIPKV_STEER_REFRESH` | `128` | Coefficient refresh interval (the reference's `divide_length`). |

`eval.py` prints the number of steered tokens and their mean coefficient, and stores them in the
result JSON.

**Validation.**
- CPU ([`tests/test_steering.py`](../../tests/test_steering.py)): the controller matches a literal
  re-implementation of the reference's per-forward flow under token-by-token decode, chunked prefill,
  preemption + recompute, prefix-cache hits and multi-request batches; `steer_residual` math. Two
  deliberate bugs (refreshing the coefficient every token; no rebuild after recompute) both fail it.
- GPU, R1-Distill-Qwen-7B, prompt ending in `".\n\n"` inside `<think>`: next-token log-probs with
  steering match a HuggingFace forward with the reference rule within bf16 noise (`Then` −0.039 vs
  −0.045; `Wait` −5.04 vs −4.80, unsteered −0.99).
- GPU decode (CUDA graphs), 300 greedy tokens with a constant coefficient: vLLM matches an HF steered
  decode loop through the first steered breaks (78 identical tokens; the unsteered run diverges at 52).
- With `coef = 0`, eager-mode output is byte-identical to no steering (300/300 tokens). Under
  torch.compile the extra add changes kernel fusion, so greedy text can drift after ~100 tokens.

