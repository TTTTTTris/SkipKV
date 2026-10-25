"""R-KV: Redundancy-aware KV Cache Compression for reasoning models.

Decoding-time KV cache compression ported onto vLLM v1 (v0.25.1), following the
same patch-style layout as the SGLang R-KV port.

This package is split into two layers:

* ``algo`` -- the device-agnostic pure algorithm (a faithful port of the
  reference ``R1KV``). It has no vLLM dependencies and can be unit-tested on
  CPU against the original.
* ``sentence`` -- SkipKV sentence-level redundancy tracking (optional, enabled
  by ``VLLM_V1_SKIPKV_SENTENCE_THRESHOLD``).
* ``integration`` -- wires the algorithm into vLLM's paged KV cache and the
  FlashAttention v1 backend (physical eviction, logical/physical position
  bookkeeping).
"""

from vllm.skipkv.algo import R1KV, cal_similarity, compute_attention_scores
from vllm.skipkv.integration import RKVCompressor, RKVConfig
from vllm.skipkv.sentence import SentenceTracker, boundary_token_ids

__all__ = [
    "R1KV",
    "cal_similarity",
    "compute_attention_scores",
    "RKVConfig",
    "RKVCompressor",
    "SentenceTracker",
    "boundary_token_ids",
]
