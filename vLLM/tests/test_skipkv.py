"""CPU tests for the SkipKV sentence-eviction extension of the vLLM R-KV port.

Covers ``skipkv/sentence.py`` (segmentation, redundancy detection, chunk/decode
equivalence, lifecycle) and the SkipKV path of ``RKVCompressor.compact_step``
(penalized sentences are evicted, logical positions survive compaction). Like
the other suites it loads the source modules by path with stub ``vllm``
packages, so no GPU or vLLM install is needed.

    python tests/test_skipkv.py
"""

import importlib.util
import os
import sys
import types
import unittest

import numpy as np
import torch

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_BASE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.modules.setdefault("vllm", types.ModuleType("vllm"))
sys.modules.setdefault("vllm.skipkv", types.ModuleType("vllm.skipkv"))
sys.modules["vllm"].__path__ = []
sys.modules["vllm.skipkv"].__path__ = []
_load("vllm.skipkv.algo", "skipkv/algo.py")
_sentence = _load("vllm.skipkv.sentence", "skipkv/sentence.py")
_integration = _load("vllm.skipkv.integration", "skipkv/integration.py")
SentenceTracker = _sentence.SentenceTracker
RKVConfig = _integration.RKVConfig
RKVCompressor = _integration.RKVCompressor

NL = 99  # boundary token id
HID = 16


def make_sequence(seed=0):
    """Token ids + hidden states whose sentence 0 is repeated as sentence 2.

    Sentences (inclusive logical ranges): (0,2), (3,7) [ends in 2 boundary
    tokens -> merged], (8,10) ~ copy of (0,2), (11,13).
    """
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([1, 2, NL, 3, 4, 5, NL, NL, 6, 7, NL, 8, 9, NL])
    h = torch.randn(len(ids), HID, generator=g)
    h[8:11] = h[0:3] + 0.01 * torch.randn(3, HID, generator=g)
    return ids, h


def feed(tracker, rid, ids, h, chunks):
    """Feed one request in consecutive chunks (sizes in ``chunks``)."""
    pos = 0
    for c in chunks:
        tracker.update(
            [rid], np.array([0, c]), np.array([pos]), ids[pos : pos + c], h[pos : pos + c]
        )
        pos += c


class TestSentenceTracker(unittest.TestCase):
    def test_segmentation_and_redundancy(self):
        ids, h = make_sequence()
        tr = SentenceTracker([NL], threshold=0.95)
        feed(tr, "a", ids, h, [len(ids)])
        self.assertEqual(tr.ranges("a"), [(0, 2), (3, 7), (8, 10), (11, 13)])
        red = tr.redundant_ranges("a")
        self.assertEqual(list(red), [(0, 2)])
        self.assertGreater(red[(0, 2)], 0.95)

    def test_decode_equals_prefill_chunks(self):
        ids, h = make_sequence()
        ref = SentenceTracker([NL], 0.95)
        feed(ref, "a", ids, h, [len(ids)])
        for chunks in ([1] * len(ids), [5, 1, 1, 7], [3, 11], [8, 6]):
            tr = SentenceTracker([NL], 0.95)
            feed(tr, "a", ids, h, chunks)
            self.assertEqual(tr.ranges("a"), ref.ranges("a"), chunks)
            self.assertEqual(tr.redundant_ranges("a").keys(), ref.redundant_ranges("a").keys())

    def test_batched_rows_are_independent(self):
        ids, h = make_sequence()
        ids2, h2 = make_sequence(seed=1)
        h2[8:11] = torch.randn(3, HID)  # request b has no repeat
        tr = SentenceTracker([NL], 0.95)
        for t in range(len(ids)):  # both requests decode in lockstep
            tr.update(
                ["a", "b"], np.array([0, 1, 2]), np.array([t, t]),
                torch.stack([ids[t], ids2[t]]), torch.stack([h[t], h2[t]]),
            )
        self.assertEqual(list(tr.redundant_ranges("a")), [(0, 2)])
        self.assertEqual(tr.redundant_ranges("b"), {})

    def test_keep_first_penalizes_the_later_copy(self):
        ids, h = make_sequence()
        tr = SentenceTracker([NL], 0.95, keep="first")
        feed(tr, "a", ids, h, [len(ids)])
        red = tr.redundant_ranges("a")
        self.assertEqual(list(red), [(8, 10)])  # the repeat, not the original (0,2)
        self.assertGreater(red[(8, 10)], 0.95)
        with self.assertRaises(ValueError):
            SentenceTracker([NL], 0.95, keep="middle")

    def test_pool_grows_past_initial_capacity(self):
        ids, h = make_sequence()
        tr = SentenceTracker([NL], 0.95)
        for n in (3, 40):  # second wave exceeds the initial 16-slot pool
            rids = [f"r{n}_{i}" for i in range(n)]
            for t in range(len(ids)):
                tr.update(
                    rids, np.arange(n + 1), np.full(n, t),
                    ids[t].repeat(n), h[t].expand(n, -1),
                )
            for rid in rids:
                self.assertEqual(list(tr.redundant_ranges(rid)), [(0, 2)])

    def test_mixed_prefill_and_decode_rows(self):
        # one request prefills in chunks while another decodes token by token
        ids, h = make_sequence()
        ref = SentenceTracker([NL], 0.95)
        feed(ref, "a", ids, h, [len(ids)])
        tr = SentenceTracker([NL], 0.95)
        chunks, pa = [4, 3, 7], 0
        for t, c in enumerate(chunks + [0] * (len(ids) - len(chunks))):
            rows, qsl, pos, tid, th = [], [0], [], [], []
            if c:
                rows.append("a"); pos.append(pa); tid.append(ids[pa : pa + c]); th.append(h[pa : pa + c])
                qsl.append(qsl[-1] + c); pa += c
            rows.append("b"); pos.append(t); tid.append(ids[t : t + 1]); th.append(h[t : t + 1])
            qsl.append(qsl[-1] + 1)
            tr.update(rows, np.array(qsl), np.array(pos), torch.cat(tid), torch.cat(th))
        for rid in ("a", "b"):
            self.assertEqual(tr.ranges(rid), ref.ranges("a"), rid)
            self.assertEqual(tr.redundant_ranges(rid).keys(), ref.redundant_ranges("a").keys())

    def test_merge_within_one_prefill_chunk(self):
        # sentence closes and a lone boundary follows inside the same chunk
        ids = torch.tensor([1, 2, NL, NL, 3, NL])
        h = torch.randn(len(ids), HID)
        for chunks in ([6], [1] * 6):
            tr = SentenceTracker([NL], 0.95)
            feed(tr, "a", ids, h, chunks)
            self.assertEqual(tr.ranges("a"), [(0, 3), (4, 5)], chunks)

    def test_lookup_after_pending_step_grows_the_pool(self):
        # A pending (deferred) step whose flush reallocates the sentence pool
        # must not be indexed with the pre-growth array.
        tr = SentenceTracker([NL], 0.0, keep="first")
        g = torch.Generator().manual_seed(0)
        tr.update(["a"], np.array([0, 3]), np.array([0]), torch.tensor([1, 2, NL]),
                  torch.randn(3, HID, generator=g))
        cap = tr._pen.shape[0]
        tr._pending = None
        # fill the pool to capacity with other requests' sentences
        rids = [f"r{i}" for i in range(cap)]
        tr.update(rids, np.arange(cap + 1), np.zeros(cap, dtype=np.int64),
                  torch.full((cap,), NL), torch.randn(cap, HID, generator=g))
        tr._flush()
        # a step for "a" that closes one more sentence is left pending (CPU: sync path,
        # so emulate by committing through update) -> pool grows past ``cap``
        tr.update(["a"], np.array([0, 2]), np.array([3]), torch.tensor([5, NL]),
                  torch.randn(2, HID, generator=g))
        self.assertGreater(tr._pen.shape[0], cap)
        out = tr.penalties("a", torch.arange(5))
        self.assertEqual(out.shape[0], 5)

    def test_penalties_by_logical_position_and_release(self):
        ids, h = make_sequence()
        tr = SentenceTracker([NL], 0.95)
        feed(tr, "a", ids, h, [len(ids)])
        pen = tr.penalties("a", torch.tensor([0, 1, 2, 3, 9, 20]))
        self.assertTrue((pen[:3] > 0.95).all())
        self.assertEqual(pen[3:].tolist(), [0.0, 0.0, 0.0])  # 20 not yet seen
        tr.release(["a"])
        self.assertIsNone(tr.penalties("a", torch.tensor([0])))
        # the freed slot is reused with a clean open-sentence sum
        feed(tr, "b", ids[:2], h[:2], [2])
        self.assertTrue(torch.allclose(tr._open_sum[tr._slot["b"]], h[:2].sum(0)))


class TestPruneEvicted(unittest.TestCase):
    """Sentences with no token left in the cache are dropped from the pool."""

    def _prune(self, tr, rid, kept):
        tr.free_evicted([tr.evicted_rows(rid, torch.as_tensor(kept))])

    def test_fully_evicted_sentence_is_dropped(self):
        ids, h = make_sequence()  # sentences (0,2) (3,7) (8,10) (11,13)
        tr = SentenceTracker([NL], 0.95)
        feed(tr, "a", ids, h, [len(ids)])
        self._prune(tr, "a", [4, 5, 12, 13])  # nothing of (0,2) or (8,10) kept
        self.assertEqual(tr.ranges("a"), [(3, 7), (11, 13)])
        self.assertEqual(tr.stats["pruned"], 2)
        # freed rows are excluded from comparisons and reused by new sentences
        self.assertEqual(int((tr._owner[: tr._used_rows] == -1).sum()), 2)
        used = tr._used_rows
        feed2 = torch.tensor([1, NL])
        tr.update(["a"], np.array([0, 2]), np.array([14]), feed2, torch.randn(2, HID))
        self.assertEqual(tr._used_rows, used)

    def test_last_sentence_never_pruned(self):
        ids, h = make_sequence()
        tr = SentenceTracker([NL], 0.95)
        feed(tr, "a", ids, h, [len(ids)])
        self._prune(tr, "a", [0])  # only position 0 kept: everything else "evicted"
        self.assertEqual(tr.ranges("a"), [(0, 2), (11, 13)])

    def test_per_head_union(self):
        ids, h = make_sequence()
        tr = SentenceTracker([NL], 0.95)
        feed(tr, "a", ids, h, [len(ids)])
        kept = torch.tensor([[[4, 12, 13], [9, 12, 13]]])  # (layers=1, heads=2, budget=3)
        self._prune(tr, "a", kept)
        self.assertEqual(tr.ranges("a"), [(3, 7), (8, 10), (11, 13)])  # (8,10) kept by head 1

    def test_prune_is_exact_for_keep_last(self):
        g = torch.Generator().manual_seed(0)
        L = 400
        ids = torch.tensor([NL if x < 0.15 else 1 for x in torch.rand(L, generator=g).tolist()])
        h = torch.randn(L, HID, generator=g)
        for r in range(0, L - 40, 60):  # plant repeats of earlier spans
            h[r + 30 : r + 40] = h[r : r + 10] + 0.01 * torch.randn(10, HID, generator=g)
            ids[r + 30 : r + 40] = ids[r : r + 10]
        on, off = SentenceTracker([NL], 0.95, prune=True), SentenceTracker([NL], 0.95, prune=False)
        kept = torch.arange(0)
        for t in range(L):
            for tr in (on, off):
                tr.update(["a"], np.array([0, 1]), np.array([t]), ids[t : t + 1], h[t : t + 1])
            if t % 25 == 24:  # a compaction: keep a random subset + the recent window
                cand = torch.cat([kept, torch.arange(max(0, t - 24), t - 7)])
                keep = cand[torch.randperm(len(cand), generator=g)[: 40]]
                kept = torch.sort(torch.cat([keep, torch.arange(t - 7, t + 1)])).values
                self._prune(on, "a", kept)
                self.assertTrue(torch.allclose(on.penalties("a", kept), off.penalties("a", kept)), t)
        self.assertGreater(on.stats["pruned"], 0)
        self.assertGreater(len(off.redundant_ranges("a")), 0)  # the planted repeats were found
        self.assertLess(len(on.ranges("a")), len(off.ranges("a")))

    def test_keep_first_ignores_evicted_originals(self):
        ids, h = make_sequence()  # (8,10) repeats (0,2)
        for prune, want in ((False, [(8, 10)]), (True, [])):
            tr = SentenceTracker([NL], 0.95, keep="first", prune=prune)
            feed(tr, "a", ids[:8], h[:8], [8])  # sentences (0,2), (3,7) closed
            self._prune(tr, "a", [4, 5, 6, 7])  # (0,2) fully evicted
            feed_rest = [(t, 1) for t in range(8, len(ids))]
            for t, c in feed_rest:
                tr.update(["a"], np.array([0, c]), np.array([t]), ids[t : t + c], h[t : t + c])
            self.assertEqual(list(tr.redundant_ranges("a")), want, prune)


def make_cache(num_slots, kv_heads=2, head_dim=4):
    """Paged cache (block_size 4) where every entry at slot s equals s."""
    vals = torch.arange(num_slots, dtype=torch.float32).view(-1, 4, 1, 1)
    return vals.expand(-1, 4, kv_heads, head_dim).contiguous()


class TestSkipKVCompaction(unittest.TestCase):
    def _compressor(self, budget, buffer, window, threshold=0.95):
        cfg = RKVConfig(
            budget=budget, buffer_size=buffer, window_size=window,
            score_mode="reference", sentence_threshold=threshold,
        )
        comp = RKVCompressor(cfg)
        comp.enable_sentences([NL])
        comp._qcount["a"] = window  # observation window already recorded
        return comp

    def _compact(self, comp, rid, score, seq_len, logical_end, k, v):
        occupied = torch.arange(seq_len)
        dropped = [0]
        comp.compact_step(
            num_reqs=1, seq_lens=torch.tensor([seq_len]),
            occupied_slot_mapping=occupied, should_compress=(True,),
            score_acc=[score], layer_caches=[(k, v, {})],
            num_dropped_tokens_list=dropped, req_ids=[rid],
            logical_ends=np.array([logical_end]),
        )
        return dropped[0]

    def test_config_flag(self):
        self.assertFalse(RKVConfig(budget=8, buffer_size=4, window_size=2).skipkv)
        self.assertTrue(
            RKVConfig(budget=8, buffer_size=4, window_size=2, sentence_threshold=0.9).skipkv
        )
        with self.assertRaises(ValueError):
            RKVConfig(budget=8, buffer_size=4, window_size=2, sentence_threshold=1.5)
        comp = RKVCompressor(RKVConfig(budget=8, buffer_size=4, window_size=2))
        comp.enable_sentences([NL])
        self.assertIsNone(comp.sentences)  # threshold 0 -> plain R-KV

    def test_redundant_sentence_is_evicted(self):
        ids, h = make_sequence()
        seq_len, budget, window = len(ids), 8, 2
        comp = self._compressor(budget, 4, window)
        feed(comp.sentences, "a", ids, h, [seq_len])
        k, v = make_cache(16), make_cache(16)
        # without the penalty the redundant sentence (0,2) scores highest
        # (R-KV scores are softmax-scale, far below a ~0.95 penalty)
        score = torch.tensor([9.0, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 8]) / 100
        self._compact(comp, "a", score, seq_len, seq_len - 1, k, v)
        kept = comp._logpos["a"].tolist()
        self.assertEqual(len(kept), budget)
        self.assertFalse({0, 1, 2} & set(kept))
        self.assertEqual(kept[-window:], [12, 13])
        # physical KV now holds exactly those logical entries, in order
        self.assertEqual([k[i // 4, i % 4, 0, 0].item() for i in range(budget)], kept)

    def test_no_redundancy_matches_plain_rkv(self):
        ids, h = make_sequence()
        h[8:11] = torch.randn(3, HID)
        score = torch.randn(len(ids) - 2)
        comp = self._compressor(8, 4, 2)
        feed(comp.sentences, "a", ids, h, [len(ids)])
        k1, v1 = make_cache(16), make_cache(16)
        self._compact(comp, "a", score.clone(), len(ids), len(ids) - 1, k1, v1)
        plain = RKVCompressor(RKVConfig(budget=8, buffer_size=4, window_size=2, score_mode="reference"))
        k2, v2 = make_cache(16), make_cache(16)
        plain.compact_step(
            num_reqs=1, seq_lens=torch.tensor([len(ids)]),
            occupied_slot_mapping=torch.arange(len(ids)), should_compress=(True,),
            score_acc=[score.clone()], layer_caches=[(k2, v2, {})],
            num_dropped_tokens_list=[0],
        )
        self.assertTrue(torch.equal(k1, k2))

    def test_logical_positions_track_across_compactions(self):
        comp = self._compressor(budget=6, buffer=4, window=2, threshold=0.99)
        k, v = make_cache(16), make_cache(16)
        # first compaction: physical == logical 0..9
        self._compact(comp, "a", torch.tensor([5.0, 0, 5, 0, 5, 0, 5, 0]), 10, 9, k, v)
        self.assertEqual(comp._logpos["a"].tolist(), [0, 2, 4, 6, 8, 9])
        # 4 new tokens (logical 10..13) appended at physical 6..9
        self._compact(comp, "a", torch.tensor([0.0, 9, 0, 9, 9, 0, 9, 0]), 10, 13, k, v)
        self.assertEqual(comp._logpos["a"].tolist(), [2, 6, 8, 10, 12, 13])
        comp.release_requests(["a"])
        self.assertNotIn("a", comp._logpos)


if __name__ == "__main__":
    unittest.main()
