"""SkipKV sentence-level redundancy tracking for the vLLM R-KV port.

SkipKV (MLSys 2026) extends R-KV's token scoring with a sentence-level
signal: the model's reasoning repeats whole sentences, and evicting a repeated
sentence *as a unit* beats fragmenting it token by token. This module is the
sentence half of that method (the activation-steering half is not ported):

* sentences are delimited by boundary tokens (newline-style tokens);
* each sentence is represented by the mean of its **final-layer hidden states**;
* when a sentence closes it is compared (cosine) against every earlier sentence
  of the same request, and each earlier sentence above ``threshold`` is marked
  redundant with penalty = similarity (the latest occurrence is kept);
* at compaction, :class:`~vllm.skipkv.integration.RKVCompressor` subtracts the
  penalty from the R-KV score of every token inside a redundant sentence.

Positions are **logical** token positions (the RoPE positions the runner
assigns), so a sentence keeps its identity across physical KV compaction.

Performance design (the tracker runs on every model step):

* **Decode steps never block on the GPU.** Hidden-state sums, sentence means
  and resets happen on the GPU; the boundary mask is copied to the host
  asynchronously and its host bookkeeping is applied at the start of the next
  step (by then the copy has long finished). Compaction therefore sees
  sentences closed up to the previous step.
* **No per-request Python.** Host state lives in numpy arrays indexed by a
  per-request slot; only the rare "lone boundary token" merge loops.
* **One batched comparison.** All sentences closed in a step are compared with
  a single ``(closed x pool)`` matmul, masked to each request's own earlier
  sentences, and the penalties are max-updated on the GPU (no readback).

Steps containing a multi-token row (prefill chunks) take a synchronous path.
It has no vLLM dependencies, so it is CPU-testable.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["SentenceTracker", "boundary_token_ids"]

# The boundary tokens used by the SkipKV / R-KV HuggingFace reference.
_BOUNDARY_STRINGS = ("\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n")


def boundary_token_ids(tokenizer) -> list[int]:
    """Sentence-boundary token ids for ``tokenizer`` (newline-style tokens)."""
    return sorted(
        {tokenizer.encode(s, add_special_tokens=False)[-1] for s in _BOUNDARY_STRINGS}
    )


def _grow(arr: np.ndarray, n: int, fill) -> np.ndarray:
    if arr.shape[0] >= n:
        return arr
    out = np.full(max(n, 2 * arr.shape[0], 16), fill, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


class SentenceTracker:
    """Per-request sentence segmentation and redundancy detection."""

    def __init__(self, boundary_ids, threshold: float, keep: str = "last", prune: bool = True):
        # ``keep``: which copy of a similar pair survives. "last" (SkipKV) scores
        # the earlier sentence; "first" scores the later one, so the prompt and
        # the first statement of a fact are protected and repeats are evicted.
        # ``prune``: drop a sentence from the pool once none of its tokens is left
        # in the KV cache, so new sentences are compared only with sentences that
        # are still cached. Exact for keep="last" (a penalty on an evicted
        # sentence has no effect); changes keep="first" (a repeat of an
        # evicted sentence is no longer penalized). Bounds the pool by what is cached.
        if keep not in ("first", "last"):
            raise ValueError(f"SentenceTracker keep must be 'first' or 'last', got {keep!r}")
        self.keep = keep
        self.prune = bool(prune)
        self.threshold = float(threshold)
        self._bids_cpu = torch.as_tensor(sorted(set(boundary_ids)), dtype=torch.long)
        self._bids: torch.Tensor | None = None
        self._device: torch.device | None = None

        # --- request slots (host) ---
        self._slot: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._num_slots = 0
        self._last_key: tuple | None = None
        self._last_slots: np.ndarray | None = None
        self._open_start = np.full(0, -1, np.int64)  # logical start of the open sentence
        self._open_len = np.zeros(0, np.int64)  # tokens in the open sentence
        self._end = np.full(0, -1, np.int64)  # last logical position fed
        self._last_end = np.full(0, -2, np.int64)  # end of the last closed sentence
        self._rows: list[list[int]] = []  # closed-sentence pool rows, in order

        # --- request slots (device) ---
        self._open_sum: torch.Tensor | None = None  # (slots, hidden) fp32
        self._open_len_g: torch.Tensor | None = None  # (slots,) fp32

        # --- closed-sentence pool ---
        self._pstart = np.zeros(0, np.int64)
        self._pend = np.zeros(0, np.int64)
        self._free_rows: list[int] = []
        self._used_rows = 0
        self._seq = 0
        self._reprs: torch.Tensor | None = None  # (rows, hidden) fp32, normalized
        self._owner: torch.Tensor | None = None  # (rows,) slot or -1
        self._seq_g: torch.Tensor | None = None  # (rows,) allocation order
        self._pen: torch.Tensor | None = None  # (rows,) penalty

        # decode step whose host bookkeeping is still pending
        self._pending = None

        self._closed = 0
        self._pruned = 0
        self._marked: torch.Tensor | None = None
        self._hist: torch.Tensor | None = None

    # ------------------------------------------------------------- storage
    def _init_device(self, hidden: torch.Tensor) -> None:
        if self._device is not None:
            return
        dev, d = hidden.device, hidden.shape[-1]
        self._device = dev
        self._bids = self._bids_cpu.to(dev)
        self._open_sum = torch.zeros(16, d, dtype=torch.float32, device=dev)
        self._open_len_g = torch.zeros(16, dtype=torch.float32, device=dev)
        self._reprs = torch.zeros(64, d, dtype=torch.float32, device=dev)
        self._owner = torch.full((64,), -1, dtype=torch.long, device=dev)
        self._seq_g = torch.zeros(64, dtype=torch.long, device=dev)
        self._pen = torch.zeros(64, dtype=torch.float32, device=dev)
        self._marked = torch.zeros((), dtype=torch.long, device=dev)
        self._hist = torch.zeros(20, dtype=torch.float32, device=dev)

    def _grow_slots(self, n: int) -> None:
        self._open_start = _grow(self._open_start, n, -1)
        self._open_len = _grow(self._open_len, n, 0)
        self._end = _grow(self._end, n, -1)
        self._last_end = _grow(self._last_end, n, -2)
        while len(self._rows) < n:
            self._rows.append([])
        if self._open_sum.shape[0] < n:
            cap = max(n, 2 * self._open_sum.shape[0])
            s = self._open_sum.new_zeros(cap, self._open_sum.shape[1])
            s[: self._open_sum.shape[0]] = self._open_sum
            ln = self._open_len_g.new_zeros(cap)
            ln[: self._open_len_g.shape[0]] = self._open_len_g
            self._open_sum, self._open_len_g = s, ln

    def _alloc_rows(self, k: int) -> np.ndarray:
        take = min(k, len(self._free_rows))
        rows = [self._free_rows.pop() for _ in range(take)]
        rows += list(range(self._used_rows, self._used_rows + k - take))
        self._used_rows += k - take
        n = self._used_rows
        self._pstart = _grow(self._pstart, n, 0)
        self._pend = _grow(self._pend, n, 0)
        if self._reprs.shape[0] < n:
            cap = max(n, 2 * self._reprs.shape[0])

            def grow(t, fill):
                out = t.new_full((cap,) + tuple(t.shape[1:]), fill)
                out[: t.shape[0]] = t
                return out

            self._reprs = grow(self._reprs, 0.0)
            self._owner = grow(self._owner, -1)
            self._seq_g = grow(self._seq_g, 0)
            self._pen = grow(self._pen, 0.0)
        return np.asarray(rows, dtype=np.int64)

    def _slots_for(self, req_ids) -> np.ndarray:
        key = tuple(req_ids)
        if key == self._last_key:
            return self._last_slots
        slots = np.empty(len(key), dtype=np.int64)
        for i, rid in enumerate(key):
            s = self._slot.get(rid)
            if s is None:
                s = self._free_slots.pop() if self._free_slots else self._num_slots
                if s == self._num_slots:
                    self._num_slots += 1
                    self._grow_slots(self._num_slots)
                self._slot[rid] = s
            slots[i] = s
        self._last_key, self._last_slots = key, slots
        return slots

    # ------------------------------------------------------------ lifecycle
    def release(self, request_ids) -> None:
        """Drop the state of finished or preempted requests (preempted ones are
        recomputed from scratch, so their sentences are re-tracked)."""
        self._flush()  # pending bookkeeping may still reference these slots
        freed_rows = []
        for rid in request_ids:
            s = self._slot.pop(rid, None)
            if s is None:
                continue
            self._free_slots.append(s)
            freed_rows += self._rows[s]
            self._rows[s] = []
            self._open_start[s], self._open_len[s] = -1, 0
            self._end[s], self._last_end[s] = -1, -2
            self._open_sum[s].zero_()
            self._open_len_g[s] = 0
        if freed_rows:
            self._free_rows += freed_rows
            idx = torch.as_tensor(freed_rows, device=self._device)
            self._owner[idx] = -1
            self._pen[idx] = 0
        self._last_key = None

    # -------------------------------------------------------------- pruning
    def evicted_rows(self, rid, kept_pos: torch.Tensor):
        """Queue the evicted-sentence check for ``rid`` after a compaction.

        ``kept_pos``: logical positions of the entries now in the cache, ``(budget,)``
        (global) or ``(layers, heads, budget)`` (per-head; a sentence is alive if any
        head keeps any of its tokens). Returns a job for :meth:`free_evicted` or None.
        The request's most recent sentence is never pruned: a following lone
        boundary token extends it (``_rows[s][-1]``). Sync-free.
        """
        if not self.prune:
            return None
        self._flush()
        s = self._slot.get(rid)
        if s is None or len(self._rows[s]) <= 1:
            return None
        rows = np.asarray(self._rows[s][:-1], dtype=np.int64)
        starts, ends = self._pstart[rows], self._pend[rows]
        top = int(ends.max())
        dev = kept_pos.device
        cover = torch.zeros(top + 2, dtype=torch.int32, device=dev)
        cover.index_fill_(0, kept_pos.reshape(-1).clamp(0, top + 1).long(), 1)
        cs = torch.cumsum(cover, 0)  # cs[i] = distinct kept positions <= i
        st = torch.from_numpy(starts).to(dev, non_blocking=True)
        en = torch.from_numpy(ends).to(dev, non_blocking=True)
        before = torch.where(st > 0, cs[(st - 1).clamp(min=0)], torch.zeros_like(cs[:1]))
        dead = (cs[en] - before) == 0  # no kept token inside [start, end]
        return s, rows, dead

    def free_evicted(self, jobs) -> None:
        """Apply queued :meth:`evicted_rows` checks (one host sync for all)."""
        jobs = [j for j in jobs if j is not None]
        if not jobs:
            return
        dead_all = torch.cat([j[2] for j in jobs]).cpu().numpy()
        o, freed = 0, []
        for s, rows, dead in jobs:
            d = dead_all[o : o + len(rows)]
            o += len(rows)
            if not d.any():
                continue
            gone = set(rows[d].tolist())
            freed += list(gone)
            self._rows[s] = [r for r in self._rows[s] if r not in gone]
        if freed:
            self._free_rows += freed
            self._pruned += len(freed)
            idx = torch.as_tensor(freed, device=self._device)
            self._owner[idx] = -1  # excluded from every comparison (owner mask)
            self._pen[idx] = 0

    # ---------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, req_ids, query_start_loc, start_positions, input_ids, hidden) -> None:
        """Feed one model step.

        ``req_ids``: request id per batch row; ``query_start_loc``: numpy
        ``(num_reqs + 1,)`` token offsets; ``start_positions``: numpy logical
        position of each row's first scheduled token; ``input_ids`` /
        ``hidden``: flat GPU tensors covering at least the scheduled tokens.
        """
        num_reqs = len(req_ids)
        if num_reqs == 0:
            return
        qsl = np.asarray(query_start_loc[: num_reqs + 1], dtype=np.int64)
        counts = np.diff(qsl)
        num_tokens = int(qsl[-1])
        if num_tokens == 0:
            return
        pos0 = np.asarray(start_positions[:num_reqs], dtype=np.int64)
        self._init_device(hidden)
        self._flush()
        slots = self._slots_for(req_ids)
        live = counts > 0
        self._end[slots[live]] = pos0[live] + counts[live] - 1

        ids = input_ids[:num_tokens]
        h = hidden[:num_tokens].float()
        if (counts == 1).all():
            self._decode_gpu(slots, pos0, ids, h, sync=False)
        else:
            self._mixed_step(slots, pos0, counts, qsl, ids, h)

    def _decode_gpu(self, slots, pos, ids, h, sync: bool):
        """One token per row: GPU accumulate/close/reset, host part deferred."""
        dev = self._device
        sl = torch.from_numpy(slots).to(dev, non_blocking=True)
        is_b = torch.isin(ids, self._bids)
        acc = self._open_sum.index_select(0, sl) + h
        n = self._open_len_g.index_select(0, sl) + 1
        means = acc / n.unsqueeze(1)  # the closing sentence's mean (where is_b)
        keep = (~is_b).float()
        self._open_sum.index_copy_(0, sl, acc * keep.unsqueeze(1))
        self._open_len_g.index_copy_(0, sl, n * keep)
        if sync or dev.type != "cuda":
            self._host_decode(slots, pos, is_b.cpu().numpy(), means)
            return
        host = torch.empty(is_b.shape, dtype=torch.bool, pin_memory=True)
        host.copy_(is_b, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._pending = (slots.copy(), pos.copy(), host, means, ev)

    def _flush(self) -> None:
        if self._pending is None:
            return
        slots, pos, host, means, ev = self._pending
        self._pending = None
        ev.synchronize()
        self._host_decode(slots, pos, host.numpy(), means)

    def _host_decode(self, slots, pos, is_b, means) -> None:
        """Host bookkeeping for a one-token-per-row step."""
        empty = self._open_len[slots] == 0
        self._open_start[slots[empty]] = pos[empty]
        self._open_len[slots] += 1
        rows = np.nonzero(is_b)[0]
        if rows.size == 0:
            return
        cs = slots[rows]
        lens = self._open_len[cs].copy()
        starts = self._open_start[cs].copy()
        ends = pos[rows]
        self._open_len[cs] = 0
        self._open_start[cs] = -1
        # a lone boundary token (consecutive punctuation) extends the previous sentence
        lone = (lens == 1) & (self._last_end[cs] == ends - 1)
        for s, e in zip(cs[lone].tolist(), ends[lone].tolist()):
            self._pend[self._rows[s][-1]] = e
        self._last_end[cs] = ends
        keep = ~lone
        if keep.any():
            sel = torch.from_numpy(rows[keep]).to(means.device, non_blocking=True)
            self._commit(cs[keep], starts[keep], ends[keep], means.index_select(0, sel))

    def _mixed_step(self, slots, pos0, counts, qsl, ids, h) -> None:
        """Synchronous path for steps with multi-token rows (prefill chunks)."""
        is_b_all = torch.isin(ids, self._bids).cpu().numpy()
        single = np.nonzero(counts == 1)[0]
        if single.size:
            tok = torch.from_numpy(qsl[single]).to(self._device)
            self._decode_gpu(slots[single], pos0[single], ids[tok], h[tok], sync=True)
        c_slots, c_starts, c_ends, c_means = [], [], [], []
        for r in np.nonzero(counts > 1)[0].tolist():
            s, base, q0, q1 = int(slots[r]), int(pos0[r]), int(qsl[r]), int(qsl[r + 1])
            seg0 = 0
            for b in np.nonzero(is_b_all[q0:q1])[0].tolist():
                seg = h[q0 + seg0 : q0 + b + 1]
                if seg0 == 0 and self._open_len[s] > 0:  # continues the open sentence
                    total = self._open_sum[s] + seg.sum(0)
                    n = int(self._open_len[s]) + b + 1
                    start = int(self._open_start[s])
                else:
                    total, n, start = seg.sum(0), b - seg0 + 1, base + seg0
                end = base + b
                self._open_sum[s].zero_()
                self._open_len[s], self._open_start[s] = 0, -1
                if n == 1 and self._last_end[s] == end - 1:
                    # extend the previous sentence: still pending in this chunk, or committed
                    if c_slots and c_slots[-1] == s:
                        c_ends[-1] = end
                    else:
                        self._pend[self._rows[s][-1]] = end
                else:
                    c_slots.append(s), c_starts.append(start), c_ends.append(end)
                    c_means.append(total / n)
                self._last_end[s] = end
                seg0 = b + 1
            tail = h[q0 + seg0 : q1]
            if tail.shape[0]:
                if self._open_len[s] == 0:
                    self._open_start[s] = base + seg0
                self._open_sum[s] += tail.sum(0)
                self._open_len[s] += tail.shape[0]
            self._open_len_g[s] = float(self._open_len[s])
        if c_slots:
            self._commit(
                np.asarray(c_slots), np.asarray(c_starts), np.asarray(c_ends),
                torch.stack(c_means),
            )

    def _commit(self, slots, starts, ends, means) -> None:
        """Add closed sentences and max-update earlier sentences' penalties."""
        k = len(slots)
        rows = self._alloc_rows(k)
        self._pstart[rows], self._pend[rows] = starts, ends
        for s, r in zip(slots.tolist(), rows.tolist()):
            self._rows[s].append(r)
        seq = np.arange(self._seq, self._seq + k, dtype=np.int64)
        self._seq += k
        self._closed += k

        dev = self._device
        rows_g = torch.from_numpy(rows).to(dev, non_blocking=True)
        slots_g = torch.from_numpy(np.ascontiguousarray(slots, dtype=np.int64)).to(dev, non_blocking=True)
        seq_g = torch.from_numpy(seq).to(dev, non_blocking=True)
        reps = means / (means.norm(dim=-1, keepdim=True) + 1e-8)
        self._reprs[rows_g] = reps
        self._owner[rows_g] = slots_g
        self._seq_g[rows_g] = seq_g
        self._pen[rows_g] = 0

        used = self._used_rows
        # (closed x pool) cosine; valid = same request and closed earlier
        sims = reps @ self._reprs[:used].T
        valid = (self._owner[:used].unsqueeze(0) == slots_g.unsqueeze(1)) & (
            self._seq_g[:used].unsqueeze(0) < seq_g.unsqueeze(1)
        )
        sims = torch.where(valid, sims, torch.full_like(sims, -1.0))
        hits = sims > self.threshold
        best = torch.where(valid.any(dim=1), sims.amax(dim=1), torch.full_like(sims[:, 0], -1.0))
        if self.keep == "last":  # credit the earlier sentence of each pair
            upd = torch.where(hits, sims, torch.zeros_like(sims)).amax(dim=0)
            self._pen[:used] = torch.maximum(self._pen[:used], upd)
        else:  # credit the just-closed (later) sentence
            self._pen[rows_g] = torch.where(best > self.threshold, best, torch.zeros_like(best))
        self._marked += hits.sum()
        self._hist += torch.histc(best, bins=20, min=0.0, max=1.0)

    # ------------------------------------------------------------- penalties
    def penalties(self, rid, logical_positions: torch.Tensor) -> torch.Tensor | None:
        """Per-entry penalty for ``logical_positions`` of request ``rid``, or
        ``None`` when the request has no closed sentence. Reflects sentences
        closed up to the previous decode step."""
        return self._per_position(rid, logical_positions, "_pen", 0.0)

    def _per_position(self, rid, logical_positions, attr, outside):
        # Flush unconditionally (compaction already syncs, so this is free and
        # keeps results timing-independent), and read the value array only
        # AFTER it: the flush can commit sentences that grow (reallocate) it.
        self._flush()
        values = getattr(self, attr)
        s = self._slot.get(rid)
        if s is None or not self._rows[s]:
            return None
        rows = np.asarray(self._rows[s], dtype=np.int64)
        starts, ends = self._pstart[rows], self._pend[rows]
        end = int(max(self._end[s], ends.max()))
        lens = ends - starts + 1
        owner = np.full(end + 1, -1, dtype=np.int64)
        owner[np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())] = np.repeat(rows, lens)
        owner_g = torch.from_numpy(owner).to(logical_positions.device, non_blocking=True)
        r = owner_g[logical_positions.clamp(max=end)]
        inside = (r >= 0) & (logical_positions <= end)
        return torch.where(inside, values[r.clamp(min=0)], torch.full_like(values[:1], outside))

    # ------------------------------------------------------ introspection
    def ranges(self, rid) -> list:
        """Closed sentences of ``rid`` as logical (start, end) (tests / debug)."""
        self._flush()
        s = self._slot.get(rid)
        return [] if s is None else [(int(self._pstart[r]), int(self._pend[r])) for r in self._rows[s]]

    def redundant_ranges(self, rid) -> dict:
        """{(start, end): penalty} for request ``rid`` (tests / debug; syncs)."""
        self._flush()
        s = self._slot.get(rid)
        if s is None or not self._rows[s]:
            return {}
        pen = self._pen[torch.as_tensor(self._rows[s], device=self._device)].tolist()
        return {rg: p for rg, p in zip(self.ranges(rid), pen) if p > 0}

    @property
    def stats(self) -> dict:
        """Sentences closed, redundant pairs marked, and a histogram of each
        closed sentence's best similarity to an earlier one (syncs)."""
        self._flush()
        if self._device is None:
            return {"closed": 0, "pruned": 0, "marked": 0, "max_sim_hist": [0] * 20}
        return {
            "closed": self._closed,
            "pruned": self._pruned,
            "marked": int(self._marked),
            "max_sim_hist": [int(v) for v in self._hist.tolist()],
        }
