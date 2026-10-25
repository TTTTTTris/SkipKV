# Documentation

The vLLM port has two layers: the **R-KV** base (token-level KV-cache compression and its wiring
into vLLM v1) and **SkipKV** on top of it (sentence-level eviction, per-head granularity and
activation steering). Both live in one package, [`skipkv/`](../skipkv/), installed by one patch.

## SkipKV — [`skipkv/`](./skipkv/)

| Doc | Contents |
| --- | --- |
| [`DESIGN.md`](./skipkv/DESIGN.md) | Design decisions: sentences across physical compaction, sync-free tracking, steering in a compiled model, recomputation after preemption, differences from the reference. |
| [`IMPLEMENTATION.md`](./skipkv/IMPLEMENTATION.md) | Details: sentence tracker, penalty scale, keep-first variant, eviction granularity, prompt diagnostics, configuration, results, validation, steering. |

## R-KV base — [`rkv/`](./rkv/)

| Doc | Contents |
| --- | --- |
| [`DESIGN.md`](./rkv/DESIGN.md) | The algorithm and the design decisions for vLLM v1. |
| [`IMPLEMENTATION.md`](./rkv/IMPLEMENTATION.md) | Per-step data flow and the exact wiring points in v0.25.1. |
| [`OPTIMIZATIONS.md`](./rkv/OPTIMIZATIONS.md) | Validation record, performance work, known limitations, roadmap. |
| [`RESIDUAL_GAP_INVESTIGATION.md`](./rkv/RESIDUAL_GAP_INVESTIGATION.md) | The accuracy-gap investigation (async-scheduling KV corruption). |

## Both

- [`REPRODUCE.md`](./REPRODUCE.md) — how the shared wiring patch was generated and how to regenerate it.
