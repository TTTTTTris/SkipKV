# SkipKV Design — vLLM port

For the *why* behind SkipKV as a method, see the paper
([MLSys 2026: SkipKV](https://openreview.net/pdf?id=0EsV9SIm8p)) and its HuggingFace reference
in the root of this repository ([`skipkv/modeling.py`](../../../skipkv/modeling.py) and
[`skipkv/compression/skipkv.py`](../../../skipkv/compression/skipkv.py)). This document focuses on the design
decisions specific to bringing SkipKV onto the **R-KV port for vLLM v1 (v0.25.1)**
([`rkv/DESIGN.md`](../rkv/DESIGN.md)). Configuration, variants, validation and results are in
[`skipkv/IMPLEMENTATION.md`](./IMPLEMENTATION.md).

## 1. The method

SkipKV adds two things to R-KV. They are independent and can be enabled separately.

**Sentence eviction** (skip redundant KV). Reasoning models repeat whole sentences; R-KV scores
tokens one at a time and leaves fragments of repeats behind. SkipKV scores sentences:

- a sentence ends at a boundary token (newline-style tokens: `"\n"`, `".\n"`, `"\n\n"`, …);
- its representation is the mean of its **final-layer hidden states**, L2-normalized;
- when it closes it is compared (cosine) with every earlier sentence of the same request; each
  earlier sentence above `threshold` (0.95) is marked redundant with penalty = similarity, so the
  latest copy survives;
- at compaction every token of a redundant sentence has `scale × penalty` subtracted from its R-KV
  score, so the sentence is evicted as a unit.

**Activation steering** (skip redundant generation). A precomputed SEAL vector `v` (the
"transition/reflection" direction at decoder layer `L = 20`) is added to the residual stream at
paragraph-break tokens (text containing `"\n\n"`) inside `<think> … </think>`:

$$h_L \mathrel{+}= \big(c_0 - \gamma \cdot n_\text{check}\big)\, v$$

where `n_check` counts sentences containing a check word (`Wait`, `Alternatively`, `again`),
refreshed every 128 positions. With `c_0 = −1.25, γ = 0.02` the reflection direction is
suppressed, more strongly the more the model has hesitated.

An ablation built on the same machinery: **keep-first** (penalize the later copy). See [`skipkv/IMPLEMENTATION.md`](./IMPLEMENTATION.md) §6.

## 2. The core tension: sentences outlive physical slots

A sentence is a range of token positions, but R-KV compaction relocates the kept KV into the
leading `budget` physical slots every `buffer` steps. A physical slot index therefore says nothing
about which token it holds, and after a few compactions a sentence's surviving tokens are scattered
over arbitrary slots.

The port identifies everything by **logical position** (the RoPE position the runner assigns,
`num_computed_tokens + offset`), never by slot:

| Component | Keyed by | Kept as |
| --- | --- | --- |
| `SentenceTracker` | logical `[start, end]` per sentence | pool of closed sentences per request |
| `RKVCompressor._logpos[req]` | physical slot → logical position | `[budget]` (global) or `[layers, kv_heads, budget]` (per-head) |

At each compaction `_logical_positions` rebuilds the map for the current cache as *kept entries*
followed by *the new tokens since the last compaction* (contiguous logical positions), the penalty
of each cached entry is looked up by logical position, and the new kept set's logical positions are
stored for the next round. A sentence thus keeps its identity however many times it is compacted.

## 3. The second tension: tracking every step without stalling the GPU

Unlike compaction (every `buffer` steps), sentence tracking runs on **every** decode step and needs
the final hidden states, which only exist after the forward. A synchronous design (read the token,
decide whether a sentence closed, compare) costs a GPU→host sync per step. The tracker instead:

- keeps running sums, sentence means and resets on the GPU;
- copies the boundary mask to pinned host memory asynchronously and applies its host bookkeeping
  **at the start of the next step**, so compaction sees sentences closed up to the previous step (at
  most one sentence per request of lag);
- compares all sentences closed in a step with one `(closed × pool)` matmul, masked to "same request,
  closed earlier", with penalties max-updated on the GPU;
- drops a sentence from the pool once none of its tokens is cached in any layer or head
  (`evicted_rows` / `free_evicted`), so the pool is bounded by what is cached, not by output length.

Prefill chunks (multi-token rows) take a synchronous path; they only occur during admission.
Invariant: `penalties()` flushes the deferred step *before* reading the pool arrays, because the
flush can grow (reallocate) them.

## 4. Penalty scale follows eviction granularity

The reference subtracts the penalty once from **each (layer, KV head)'s** score and takes a
separate top-k per (layer, head). The port supports both granularities with one rule: subtract the
penalty at the same scale as the score it is ranked against.

- **global** (default, as the SGLang port): one kept set; the score is the per-layer head-mean
  summed over layers, so `scale = num_layers`.
- **head** (`VLLM_V1_SKIPKV_GRANULARITY=head`, as the reference): each (layer, head) ranks its own
  score, so `scale = 1`. Under tensor parallelism each rank owns its heads, so the cross-rank score
  all-reduce is skipped.

R-KV's shared paged layout ([`rkv/DESIGN.md`](../rkv/DESIGN.md) §2) only requires every layer and
head to keep the same **number** of entries, not the same tokens. Both granularities keep exactly
`budget` entries per (layer, head), so `seq_len`, the block table and `num_dropped_tokens` stay
shared per request; per-head only changes which token sits in each slot. It is much more accurate
when the budget is small relative to the prompt ([`IMPLEMENTATION.md`](./IMPLEMENTATION.md) §7).

## 5. Steering inside a compiled, graph-captured model

vLLM compiles the model forward with torch.compile and replays CUDA graphs, so Python hooks with
per-step decisions would run once at capture and never again. The port splits steering into a
graph-safe tensor op and a host-side planner:

- **In the model** (`Qwen2Model.forward`, via the patch): before layer `L`,
  `residual += coef[:n, None] · v`. vLLM layers carry the residual stream as
  `hidden_states + residual`, so this equals the reference's `hidden_states += coef · v`. `v` and
  `coef` (one fp32 entry per token, 0 = not steered) are persistent buffers on the model, so captured
  graphs read each step's values.
- **On the host** (`SteeringController.plan`, once per step in `_prepare_inputs`): decide which
  scheduled tokens are steered and with what coefficient, from the token ids the runner already
  holds, and copy the coefficients to the buffer. No GPU sync.

Three consequences are enforced in config rather than documented:

- **synchronous scheduling** — under async scheduling the host only has a step's token ids one
  step late;
- **the V1 model runner** — vLLM v0.25.1 defaults to the V2 runner when R-KV is off (Full-KV +
  steering), and the wiring lives in V1;
- **the compile cache key** — `VLLM_V1_SKIPKV_STEER_VECTOR` and `_LAYER` are registered in
  `vllm/envs.py`, so turning steering on or moving it produces a different cached graph. Without
  this, a graph compiled with steering off was reused and steering silently did nothing.

The coefficient, γ and the refresh interval are runtime buffer values and deliberately do not
invalidate the cache.

## 6. Per-step data flow

```
_prepare_inputs                      token ids gathered on the host
  → SteeringController.plan          → coef buffer (GPU)                    [steering]
model forward
  → layer L: residual += coef · v                                           [steering]
  → attention: R-KV observe_layer (scores) + register layer caches          [R-KV]
SentenceTracker.update(final hidden states)                                 [sentences]
RKVCompressor.compact_step (armed requests over budget + buffer)
  → _logical_positions → SentenceTracker.penalties → score − scale·penalty  [sentences]
  → top-k (global or per head) → relocate KV → publish num_dropped_tokens   [R-KV]
  → evicted_rows / free_evicted (prune the sentence pool)                   [sentences]
```

## 7. Recomputation (preemption and prefix-cache hits)

A preempted request loses its KV and is recomputed from scratch, often in a single multi-token step.
Each component handles it from first principles:

- **Sentences**: `reset_request_windows` drops the request's tracker state and `_logpos`; the replay
  re-segments the sequence.
- **Compaction timing**: R-KV and SkipKV wait for `window_size` genuine decode steps after a resume
  before compacting (their observation window was reset).
- **Steering**: its per-request state is a pure function of the token prefix, so a request whose
  first scheduled position differs from where its state ended is rebuilt from the token table and
  its replayed positions are steered exactly as the first time.

## 8. Deviations from the reference

| Aspect | Reference | This port |
| --- | --- | --- |
| Eviction granularity | per (layer, KV head) | global by default; per-head via `VLLM_V1_SKIPKV_GRANULARITY=head` |
| Steering coefficient / length counter | one per batch | one per request (equal at batch size 1) |
| Sentence boundaries (eviction) | includes `"?\n\n"` | omits `"?\n\n"`; steering uses the reference's set |
| Sentence pool | kept until the request ends | pruned when no token is cached (exact for keep-last) |
| Models with steering | Llama, Qwen2, Qwen3 | Qwen2 architecture (`qwen2.py`) |

See [`skipkv/IMPLEMENTATION.md`](./IMPLEMENTATION.md) for configuration (§9, §13), results (§10) and validation (§11, §13),
and [`rkv/IMPLEMENTATION.md`](../rkv/IMPLEMENTATION.md) for the exact code sites.
