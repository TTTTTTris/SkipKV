"""SkipKV activation steering for the vLLM R-KV port.

SkipKV (MLSys 2026) pairs sentence eviction with SEAL-style steering of the
model's latent state, so that it generates fewer redundant "reflection"
sentences in the first place. This module ports that half; it is independent of
KV compression (it also runs with R-KV off, i.e. Full-KV + steering).

The method (as in the SkipKV reference, ``skipkv/modeling.py``):

* a precomputed vector ``v`` (SEAL's "transition/reflection" direction at
  decoder layer ``L``) is added to the residual stream entering layer ``L``:
  ``h += coef * v``;
* only at **paragraph-break tokens** -- tokens whose text contains "\\n\\n" --
  and only while the request is inside its ``<think> ... </think>`` block;
* the coefficient grows with how often the model has re-checked itself:
  ``coef = coef_init - gamma * n_check``, where ``n_check`` counts the
  sentences that contain a check word ("Alternatively", "Wait", "again");
  sentences end at newline-style tokens. With a negative ``coef_init`` (SkipKV
  uses -1.25, gamma 0.02, layer 20 for R1-Distill-Qwen-7B) the reflection
  direction is suppressed, more strongly the more the model hesitates;
* ``coef`` is refreshed on the reference's compression steps, not every token:
  after the token at position ``p`` when ``(p + 1) % refresh == 0`` (its default
  ``divide_method="step_length"``, ``divide_length=128``; ``p`` counts prompt
  tokens too), and is used from the next token on.

The reference keeps one coefficient and one length counter per batch; here both
are per request, which equals the reference at batch size 1.

Runtime split:

* :class:`SteeringController` (host, once per step): decides which scheduled
  tokens are steered and with what coefficient, from the token ids the runner
  already has on the host, and writes one coefficient per token into a GPU
  buffer (0 = not steered).
* :func:`steer_residual` (inside the model forward, CUDA-graph safe): adds
  ``coef[:, None] * v`` to the residual before layer ``L``. The buffer and the
  vector are registered on the model, so captured graphs see the per-step
  values.

Per-request state is a deterministic function of the token prefix, so a request
that is recomputed (preemption, prefix-cache hit) is re-scanned from its tokens
and its replayed positions are steered exactly as the first time.

It has no vLLM dependencies, so it is CPU-testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

CHECK_WORDS = ("Alternatively", "Wait", "again")
# Sentence ends (the reference's ``newline_token_ids``).
BOUNDARY_STRINGS = ("\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n", "?\n\n")
THINK_START, THINK_END = "<think>", "</think>"


@dataclass
class SteeringConfig:
    vector_path: str
    layer: int = 20
    coef: float = -1.25
    gamma: float = 0.02
    refresh: int = 128  # coefficient refresh interval (the reference's divide_length)

    @classmethod
    def from_env(cls) -> SteeringConfig | None:
        """``VLLM_V1_SKIPKV_STEER_VECTOR`` (path) enables steering; ``_LAYER``,
        ``_COEF``, ``_GAMMA`` and ``_REFRESH`` default to SkipKV's
        R1-Distill-Qwen-7B settings."""
        path = os.getenv("VLLM_V1_SKIPKV_STEER_VECTOR", "")
        if not path:
            return None
        return cls(
            vector_path=path,
            layer=int(os.getenv("VLLM_V1_SKIPKV_STEER_LAYER", "20")),
            coef=float(os.getenv("VLLM_V1_SKIPKV_STEER_COEF", "-1.25")),
            gamma=float(os.getenv("VLLM_V1_SKIPKV_STEER_GAMMA", "0.02")),
            refresh=max(1, int(os.getenv("VLLM_V1_SKIPKV_STEER_REFRESH", "128"))),
        )


def steering_token_ids(tokenizer) -> dict:
    """Token-id sets the controller needs, derived as in the reference."""
    # Paragraph breaks: every vocab token whose byte-level text contains "\n\n"
    # ("ĊĊ" in GPT-2-style byte BPE; the reference matches the same way).
    split = sorted(i for t, i in tokenizer.get_vocab().items() if "ĊĊ" in t or "\n\n" in t)
    last = lambda s: tokenizer.encode(s, add_special_tokens=False)[-1]  # noqa: E731
    return {
        "split": split,
        "check": sorted({last(w) for w in CHECK_WORDS}),
        "boundary": sorted({last(s) for s in BOUNDARY_STRINGS}),
        "think_start": last(THINK_START),
        "think_end": last(THINK_END),
    }


def _lookup(ids, size: int) -> np.ndarray:
    m = np.zeros(size, dtype=bool)
    m[np.asarray(list(ids), dtype=np.int64)] = True
    return m


def steer_residual(model, hidden_states: torch.Tensor, residual: torch.Tensor | None):
    """Add ``coef * v`` to the residual stream entering the steering layer.

    vLLM decoder layers carry the stream as ``hidden_states + residual``, so adding
    to ``residual`` equals the reference's ``hidden_states += coef * v`` before the
    layer. (Before layer 0 there is no residual yet; then add to the hidden state.)
    The product is formed in fp32 and cast once, as the reference does.
    """
    n = hidden_states.shape[0]
    delta = (model.rkv_steer_coef[:n, None] * model.rkv_steer_vec).to(hidden_states.dtype)
    if residual is None:
        return hidden_states + delta, residual
    return hidden_states, residual + delta


class SteeringController:
    """Host-side per-step planning of which tokens to steer, and how strongly."""

    def __init__(self, config: SteeringConfig, token_ids: dict, vocab_size: int,
                 max_num_tokens: int, device="cpu"):
        self.config = config
        size = max(vocab_size, 1 + max(max(token_ids["split"], default=0),
                                       max(token_ids["check"]), max(token_ids["boundary"]),
                                       token_ids["think_start"], token_ids["think_end"]))
        self.is_split = _lookup(token_ids["split"], size)
        self.is_check = _lookup(token_ids["check"], size)
        self.is_bnd = _lookup(token_ids["boundary"], size)
        self.think_start = token_ids["think_start"]
        self.think_end = token_ids["think_end"]
        pin = torch.device(device).type == "cuda"
        self.coef_host = torch.zeros(max_num_tokens, dtype=torch.float32, pin_memory=pin)
        self.coef_gpu = torch.zeros(max_num_tokens, dtype=torch.float32, device=device)
        self._coef_np = self.coef_host.numpy()
        self._hi = 0  # entries of coef_gpu that may be nonzero
        # per-request state, indexed by a slot (requests come and go)
        self._slot: dict[str, int] = {}
        self._free: list[int] = []
        self.think = np.zeros(16, dtype=bool)
        self.has_check = np.zeros(16, dtype=bool)
        self.n_check = np.zeros(16, dtype=np.int64)
        self.snap = np.zeros(16, dtype=np.int64)  # n_check as of the last refresh
        # The reference labels the whole prompt in one forward and records only
        # the first check word in it; later prompt check words are ignored.
        self.prompt_check = np.zeros(16, dtype=bool)
        self.next_pos = np.zeros(16, dtype=np.int64)  # first position not yet scanned
        self.stats = {"steered_tokens": 0, "coef_sum": 0.0}

    # ---- request lifecycle -------------------------------------------------
    def _get_slot(self, rid: str) -> int:
        s = self._slot.get(rid)
        if s is None:
            s = self._free.pop() if self._free else len(self._slot)
            if s >= self.think.shape[0]:
                n = max(2 * self.think.shape[0], s + 1)
                for name in ("think", "has_check", "n_check", "snap", "prompt_check", "next_pos"):
                    a = getattr(self, name)
                    b = np.zeros(n, dtype=a.dtype)
                    b[: a.shape[0]] = a
                    setattr(self, name, b)
            self._slot[rid] = s
            self._reset(s)
        return s

    def _reset(self, s: int) -> None:
        self.think[s] = self.has_check[s] = self.prompt_check[s] = False
        self.n_check[s] = self.snap[s] = self.next_pos[s] = 0

    def release(self, request_ids) -> None:
        for rid in request_ids:
            s = self._slot.pop(rid, None)
            if s is not None:
                self._free.append(s)

    # ---- per-token rule (shared by the vectorized and the scan path) --------
    def _scan(self, s: int, toks: np.ndarray, pos0: int, prompt_len: int, out=None) -> int:
        """Advance slot ``s`` over ``toks`` (positions ``pos0...``); write the
        coefficient of each steered token into ``out`` if given. Returns #steered."""
        c0, g, R = self.config.coef, self.config.gamma, self.config.refresh
        think, has_check = bool(self.think[s]), bool(self.has_check[s])
        n_check, snap = int(self.n_check[s]), int(self.snap[s])
        pcheck = bool(self.prompt_check[s])
        steered = 0
        for j, t in enumerate(toks.tolist()):
            p = pos0 + j
            think = (think or t == self.think_start) and t != self.think_end
            if out is not None and think and self.is_split[t] and p >= prompt_len - 1:
                out[j] = c0 - g * snap
                steered += 1
            if self.is_check[t] and (p >= prompt_len or not pcheck):
                has_check = True
                pcheck |= p < prompt_len
            if self.is_bnd[t]:
                n_check += has_check
                has_check = False
            if p >= prompt_len - 1 and (p + 1) % R == 0:  # a compression step
                snap = n_check
        self.think[s], self.has_check[s] = think, has_check
        self.n_check[s], self.snap[s], self.prompt_check[s] = n_check, snap, pcheck
        self.next_pos[s] = pos0 + len(toks)
        return steered

    # ---- one model step ------------------------------------------------------
    def plan(self, req_ids, num_computed, num_scheduled, prompt_lens,
             input_ids, token_ids, cu_num_tokens) -> int:
        """Fill the coefficient buffer for this step's flattened tokens.

        ``input_ids``: this step's scheduled token ids (host, flattened);
        ``token_ids``: the batch's full token table (host, ``[row, position]``),
        used to rebuild a request's state when it is recomputed;
        ``cu_num_tokens``: cumulative scheduled tokens per row.
        Returns the number of tokens steered this step.
        """
        coef = self._coef_np
        coef[: self._hi] = 0.0
        hi, steered = 0, 0
        n = len(req_ids)
        slots = np.fromiter((self._get_slot(r) for r in req_ids), dtype=np.int64, count=n)
        num_computed = np.asarray(num_computed[:n], dtype=np.int64)
        num_scheduled = np.asarray(num_scheduled[:n], dtype=np.int64)
        ends = np.asarray(cu_num_tokens[:n], dtype=np.int64)
        starts = ends - num_scheduled
        fast = (num_scheduled == 1) & (self.next_pos[slots] == num_computed)

        # Decode rows: one token each, state in sync -- vectorized.
        if fast.any():
            rows, s = np.nonzero(fast)[0], slots[fast]
            tok = np.asarray(input_ids[starts[rows]], dtype=np.int64)
            think = (self.think[s] | (tok == self.think_start)) & (tok != self.think_end)
            hit = think & self.is_split[tok]  # a decode token is never inside the prompt
            if hit.any():
                idx = starts[rows][hit]
                coef[idx] = self.config.coef - self.config.gamma * self.snap[s][hit]
                steered += int(hit.sum())
                hi = max(hi, int(idx.max()) + 1)
            has_check = self.has_check[s] | self.is_check[tok]
            bnd = self.is_bnd[tok]
            self.n_check[s] += has_check & bnd
            self.has_check[s] = has_check & ~bnd
            self.think[s] = think
            pos = num_computed[fast]
            refresh = (pos + 1) % self.config.refresh == 0
            if refresh.any():
                self.snap[s[refresh]] = self.n_check[s[refresh]]
            self.next_pos[s] = pos + 1

        # Everything else (prefill chunks, recomputed / out-of-sync requests).
        for i in np.nonzero(~fast)[0].tolist():
            s, c, k = int(slots[i]), int(num_computed[i]), int(num_scheduled[i])
            if self.next_pos[s] != c:  # recomputed: rebuild from the token prefix
                self._reset(s)
                if c:
                    self._scan(s, np.asarray(token_ids[i, :c]), 0, int(prompt_lens[i]))
            a = int(starts[i])
            got = self._scan(s, np.asarray(input_ids[a:a + k]), c, int(prompt_lens[i]),
                             out=coef[a:a + k])
            if got:
                steered += got
                hi = max(hi, a + k)

        if steered:
            self.stats["steered_tokens"] += steered
            self.stats["coef_sum"] += float(coef[:hi].sum())
        upto = max(hi, self._hi)
        if upto:  # also clears last step's entries
            self.coef_gpu[:upto].copy_(self.coef_host[:upto], non_blocking=True)
        self._hi = hi
        return steered
