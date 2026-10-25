"""CPU tests for ``granularity='head'`` (per-layer, per-KV-head eviction).

Checks the vLLM compaction against the HuggingFace reference semantics: every
(layer, KV head) keeps its own top ``budget - window`` tokens by the R-KV score
of that layer/head (``R1KV._scores``, as ``update_kv`` does), plus the trailing
window. Also covers SkipKV's per-(layer, head) logical positions across two
compactions. Loads the source modules with stub ``vllm`` packages (no GPU).

    python tests/test_rkv_per_head.py
"""

import importlib.util
import os
import sys
import types
import unittest

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
_load("vllm.skipkv.sentence", "skipkv/sentence.py")
_integration = _load("vllm.skipkv.integration", "skipkv/integration.py")
RKVConfig = _integration.RKVConfig
RKVCompressor = _integration.RKVCompressor

BS, H, D, LAYERS = 4, 2, 8, 3
BUDGET, BUFFER, WINDOW = 6, 4, 2
NL = 99


def make_layers(num_blocks, seed):
    """Per layer: random keys; values tag the source slot and head
    (value[slot, h, :] = 10 * slot + h), so relocated entries are identifiable."""
    g = torch.Generator().manual_seed(seed)
    layers = []
    slots = torch.arange(num_blocks * BS, dtype=torch.float32).view(num_blocks, BS, 1, 1)
    heads = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    for _ in range(LAYERS):
        k = torch.randn(num_blocks, BS, H, D, generator=g)
        v = (10 * slots + heads).expand(num_blocks, BS, H, D).contiguous()
        layers.append((k, v))
    return layers


def run_step(comp, layers, occupied, rid, logical_end=None, seed=0, prompt_lens=None):
    """Record a full observation window, score every layer, compact once."""
    g = torch.Generator().manual_seed(seed)
    seq_len = occupied.numel()
    md = types.SimpleNamespace(
        num_reqs=1, should_compress_list=(True,), occupied_slot_mapping=occupied,
        seq_lens=torch.tensor([seq_len]), query_start_loc=torch.tensor([0, 1]),
        rkv_score_acc=[None], rkv_layer_caches=[], rkv_req_ids=[rid],
    )
    for _ in range(WINDOW):  # fill the query ring (same query per layer is fine)
        md.rkv_qplan = comp.plan_qwrite([rid], torch.device("cpu"))
        comp.record_query(torch.randn(1, H, D, generator=g), md)
    for k, v in layers:
        comp.observe_layer(torch.zeros(1, H, D), k, v, md)
    dropped = [0]
    comp.compact_step(
        num_reqs=1, seq_lens=md.seq_lens, occupied_slot_mapping=occupied,
        should_compress=(True,), score_acc=md.rkv_score_acc,
        layer_caches=md.rkv_layer_caches, num_dropped_tokens_list=dropped,
        req_ids=[rid] if logical_end is not None else None,
        logical_ends=None if logical_end is None else [logical_end],
        prompt_lens=prompt_lens,
    )
    return dropped[0]


def expected_kept(comp, layers_before, occupied):
    """(LAYERS, H, BUDGET) kept indices per the HF reference: per (layer, head)
    top-k of R1KV._scores over the recorded window queries, plus the window."""
    seq_len = occupied.numel()
    blk, off = occupied // BS, occupied % BS
    q = comp._qring[:, comp._slot["r"]]  # (window, q_heads, D)
    out = []
    for k, _ in layers_before:
        keys = k[blk, off].permute(1, 0, 2).unsqueeze(0)  # (1, H, seq, D)
        sc = comp.algo._scores(keys, q.permute(1, 0, 2).unsqueeze(0))[0]  # (H, P)
        top = sc.topk(BUDGET - WINDOW, dim=-1).indices
        win = torch.arange(seq_len - WINDOW, seq_len).expand(H, WINDOW)
        out.append(torch.sort(torch.cat([top, win], -1), -1).values)
    return torch.stack(out)


class TestPerHeadGranularity(unittest.TestCase):
    def _comp(self, **kw):
        return RKVCompressor(RKVConfig(
            budget=BUDGET, buffer_size=BUFFER, window_size=WINDOW, kernel_size=1,
            score_mode="batched", granularity="head", **kw))

    def test_matches_hf_per_head_topk(self):
        occupied = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7, 20, 21])  # blocks 3,1,5
        layers = make_layers(6, seed=1)
        before = [(k.clone(), v.clone()) for k, v in layers]
        comp = self._comp()
        self.assertEqual(run_step(comp, layers, occupied, "r"), occupied.numel() - BUDGET)
        want = expected_kept(comp, before, occupied)
        # the test only means something if heads/layers really disagree
        self.assertGreater(len({tuple(x.tolist()) for x in want.reshape(-1, BUDGET)}), 1)
        for l, (_, v) in enumerate(layers):
            for h in range(H):
                for j in range(BUDGET):
                    dst = occupied[j].item()
                    src = occupied[want[l, h, j]].item()
                    got = v[dst // BS, dst % BS, h, 0].item()
                    self.assertEqual(got, 10 * src + h, f"layer {l} head {h} slot {j}")

    def test_global_mode_unchanged_single_kept_set(self):
        occupied = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7, 20, 21])
        layers = make_layers(6, seed=1)
        comp = RKVCompressor(RKVConfig(
            budget=BUDGET, buffer_size=BUFFER, window_size=WINDOW, kernel_size=1,
            score_mode="batched"))
        run_step(comp, layers, occupied, "r")
        tags = {
            tuple(int(layers[l][1][occupied[j] // BS, occupied[j] % BS, h, 0].item()) // 10
                  for j in range(BUDGET))
            for l in range(LAYERS) for h in range(H)
        }
        self.assertEqual(len(tags), 1)  # one kept set for every layer and head

    def test_skipkv_logical_positions_per_head_across_compactions(self):
        occupied = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7, 20, 21])
        seq_len = occupied.numel()
        layers = make_layers(6, seed=2)
        before = [(k.clone(), v.clone()) for k, v in layers]
        comp = self._comp(sentence_threshold=0.95)
        comp.enable_sentences([NL])
        run_step(comp, layers, occupied, "r", logical_end=seq_len - 1)
        lp = comp._logpos["r"]
        self.assertEqual(tuple(lp.shape), (LAYERS, H, BUDGET))
        self.assertTrue(torch.equal(lp, expected_kept(comp, before, occupied)))
        # BUFFER new tokens (logical seq_len..seq_len+3) land after the kept entries
        run_step(comp, layers, occupied, "r", logical_end=seq_len + BUFFER - 1, seed=3)
        lp2 = comp._logpos["r"]
        self.assertEqual(tuple(lp2.shape), (LAYERS, H, BUDGET))
        # every kept position is either an earlier kept one or one of the new tokens
        for l in range(LAYERS):
            for h in range(H):
                allowed = set(lp[l, h].tolist()) | set(range(seq_len, seq_len + BUFFER))
                self.assertTrue(set(lp2[l, h].tolist()) <= allowed)
                self.assertEqual(lp2[l, h, -WINDOW:].tolist(),
                                 list(range(seq_len + BUFFER - WINDOW, seq_len + BUFFER)))

    def test_skipkv_penalty_lookup_after_per_head_compaction(self):
        # a closed, repeated sentence makes penalties() return a tensor on the
        # SECOND compaction, when logpos is (layers, heads, seq)
        occupied = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7, 20, 21])
        seq_len = occupied.numel()
        layers = make_layers(6, seed=4)
        comp = self._comp(sentence_threshold=0.5)
        comp.enable_sentences([NL])
        ids = torch.tensor([1, 2, NL, 1, 2, NL, 3, 4, 5, 6, 7, 8, 9, 10])
        hid = torch.randn(len(ids), 16)
        hid[3:6] = hid[0:3]  # sentence (3,5) repeats (0,2)
        comp.sentences.update(["r"], torch.tensor([0, len(ids)]).numpy(),
                              torch.tensor([0]).numpy(), ids, hid)
        run_step(comp, layers, occupied, "r", logical_end=seq_len - 1)
        pen = comp.sentences.penalties("r", comp._logpos["r"])
        self.assertEqual(tuple(pen.shape), (LAYERS, H, BUDGET))
        run_step(comp, layers, occupied, "r", logical_end=seq_len + BUFFER - 1, seed=3)
        self.assertEqual(tuple(comp._logpos["r"].shape), (LAYERS, H, BUDGET))

    def test_prompt_stats(self):
        import os as _os
        occupied = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7, 20, 21])
        seq_len, P = occupied.numel(), 4  # first 4 logical positions are the prompt
        for gran in ("head", "global"):
            layers = make_layers(6, seed=5)
            before = [(k.clone(), v.clone()) for k, v in layers]
            _os.environ["VLLM_V1_SKIPKV_PROMPT_STATS"] = "1"
            comp = RKVCompressor(RKVConfig(budget=BUDGET, buffer_size=BUFFER, window_size=WINDOW,
                                           kernel_size=1, score_mode="batched", granularity=gran))
            del _os.environ["VLLM_V1_SKIPKV_PROMPT_STATS"]
            md_dropped = run_step(comp, layers, occupied, "r", logical_end=seq_len - 1, prompt_lens=[P])
            st = comp.prompt_stats
            kept = comp._logpos["r"].reshape(-1, BUDGET)
            want_avg = ((kept < P).sum(-1).float() / P).mean().item()
            want_any = len({int(x) for x in kept.reshape(-1).tolist() if x < P}) / P
            want_distinct = len(set(kept.reshape(-1).tolist())) / BUDGET
            self.assertEqual(st["events"], 1, gran)
            self.assertAlmostEqual(st["prompt_kept_per_head"], want_avg, places=5)
            self.assertAlmostEqual(st["prompt_kept_any_head"], want_any, places=5)
            self.assertAlmostEqual(st["distinct_over_budget"], want_distinct, places=5)
            if gran == "global":
                self.assertAlmostEqual(st["distinct_over_budget"], 1.0)
                self.assertAlmostEqual(st["prompt_kept_per_head"], st["prompt_kept_any_head"])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            RKVConfig(budget=8, buffer_size=4, window_size=2, granularity="layer")
        with self.assertRaises(ValueError):
            RKVConfig(budget=8, buffer_size=4, window_size=2, granularity="head",
                      score_mode="reference")


if __name__ == "__main__":
    unittest.main()
