"""CPU tests for SkipKV activation steering (skipkv/steering.py).

The controller is checked against a literal re-implementation of the SkipKV
reference's per-forward flow (skipkv/modeling.py at batch size 1: one prefill
forward over the prompt, then one forward per generated token; label sentences,
steer the last token, then refresh the coefficient every ``divide_length``
processed tokens), under the schedules vLLM produces: token-by-token decode,
chunked prefill, preemption + recompute, and several requests per step.
"""

import importlib.util
import os
import random
import sys
import types
import unittest

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rkv_steering", os.path.join(_HERE, "..", "skipkv", "steering.py"))
steering = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = steering  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(steering)

# Synthetic vocabulary: 0..39 ordinary tokens, then the special ones.
NN, N, DOT_NN, COLON_NN, WAIT, AGAIN, THINK, END_THINK = range(40, 48)
VOCAB = 48
IDS = {
    "split": [NN, DOT_NN, COLON_NN],          # contain "\n\n"
    "check": [WAIT, AGAIN],
    "boundary": [N, NN, DOT_NN],              # COLON_NN is not a sentence end
    "think_start": THINK,
    "think_end": END_THINK,
}
C0, G, R = -1.25, 0.02, 8  # small refresh interval so tests cross many refreshes


def reference(tokens, prompt_len, c0=C0, g=G, refresh=R):
    """{position: coef} steered by the SkipKV reference (batch size 1)."""
    check, punct, split = set(IDS["check"]), set(IDS["boundary"]), set(IDS["split"])
    coef, think, length = c0, None, 0
    check_pos, labels, last_start = [], [], 0
    out = {}
    forwards = [list(range(prompt_len))] + [[p] for p in range(prompt_len, len(tokens))]
    for f in forwards:
        for p in f:  # check_punct_tokens_in_forward: first check word per forward
            if tokens[p] in check:
                check_pos.append(p)
                break
        for p in f:
            if tokens[p] in punct:
                rng = (last_start, p)
                last_start = p + 1
                labels.append(any(rng[0] <= x <= rng[1] for x in check_pos))
        if think is None:  # new round: does the prompt contain <think>?
            think = any(tokens[p] == THINK for p in f)
        last = tokens[f[-1]]
        think = (think or last == THINK) and last != END_THINK
        if think and last in split:
            out[f[-1]] = coef
        length += len(f)
        if length % refresh == 0:  # divide_method="step_length"
            coef = c0 - g * sum(labels)
    return out


def random_sequence(rng, prompt_len, gen_len):
    """Prompt ending in <think>, then a generation rich in the special tokens."""
    weights = [1] * 40 + [6, 4, 3, 2, 3, 2, 0, 0]
    prompt = [rng.randrange(40) for _ in range(prompt_len - 1)] + [THINK]
    if prompt_len > 4 and rng.random() < 0.5:  # check words inside the prompt too
        prompt[1], prompt[2] = WAIT, N
    gen = rng.choices(range(VOCAB), weights=weights, k=gen_len)
    if rng.random() < 0.5:
        gen[gen_len * 3 // 4] = END_THINK
    return prompt + gen


def run(controller, seqs, prompt_lens, schedule):
    """Drive ``controller`` like the model runner. ``schedule`` is a list of steps,
    each {req: (num_computed, num_scheduled)}; returns {req: {pos: coef}}."""
    maxlen = max(len(s) for s in seqs.values())
    got = {r: {} for r in seqs}
    for step in schedule:
        reqs = list(step)
        table = np.zeros((len(reqs), maxlen), dtype=np.int64)
        flat, cu = [], []
        for i, r in enumerate(reqs):
            table[i, : len(seqs[r])] = seqs[r]
            c, k = step[r]
            flat += seqs[r][c:c + k]
            cu.append(len(flat))
        controller.plan(reqs, np.array([step[r][0] for r in reqs]),
                        np.array([step[r][1] for r in reqs]),
                        np.array([prompt_lens[r] for r in reqs]),
                        np.array(flat), table, np.array(cu))
        coef = controller.coef_gpu.numpy()
        for i, r in enumerate(reqs):
            c, k = step[r]
            a = cu[i] - k
            for j in range(k):
                if coef[a + j] != 0:
                    got[r][c + j] = float(coef[a + j])
        assert not coef[len(flat):].any(), "stale coefficients past this step's tokens"
    return got


def make_controller(refresh=R, max_tokens=4096):
    cfg = steering.SteeringConfig(vector_path="", layer=1, coef=C0, gamma=G, refresh=refresh)
    return steering.SteeringController(cfg, IDS, VOCAB, max_tokens)


def assert_same(tc, got, want):
    tc.assertEqual(sorted(got), sorted(want))
    for p in want:
        tc.assertAlmostEqual(got[p], want[p], places=5)


class TestSteeringController(unittest.TestCase):
    def test_decode_matches_reference(self):
        rng = random.Random(0)
        for trial in range(20):
            pl = rng.randrange(3, 20)
            seq = random_sequence(rng, pl, rng.randrange(20, 120))
            ctl = make_controller()
            sched = [{"a": (0, pl)}] + [{"a": (p, 1)} for p in range(pl, len(seq))]
            got = run(ctl, {"a": seq}, {"a": pl}, sched)["a"]
            assert_same(self, got, reference(seq, pl))

    def test_coefficient_schedule(self):
        # 2 check sentences closed before the refresh at position 7 -> later
        # paragraph breaks get c0 - 2g; the one at position 5 still gets c0.
        seq = [1, THINK, WAIT, N, AGAIN, NN, 2, 3, 4, NN]
        ctl = make_controller()
        sched = [{"a": (0, 2)}] + [{"a": (p, 1)} for p in range(2, len(seq))]
        got = run(ctl, {"a": seq}, {"a": 2}, sched)["a"]
        self.assertEqual(sorted(got), [5, 9])
        self.assertAlmostEqual(got[5], C0)
        self.assertAlmostEqual(got[9], C0 - 2 * G)
        assert_same(self, got, reference(seq, 2))

    def test_no_steering_outside_think(self):
        seq = [1, 2, NN, 3, THINK, NN, END_THINK, NN, 4, NN]
        ctl = make_controller()
        sched = [{"a": (0, 4)}] + [{"a": (p, 1)} for p in range(4, len(seq))]
        got = run(ctl, {"a": seq}, {"a": 4}, sched)["a"]
        self.assertEqual(sorted(got), [5])  # only the break between <think> and </think>

    def test_chunked_prefill_and_preemption_replay(self):
        rng = random.Random(1)
        for trial in range(20):
            pl = rng.randrange(5, 40)
            seq = random_sequence(rng, pl, rng.randrange(30, 150))
            want = reference(seq, pl)
            sched, c = [], 0
            while c < pl:  # chunked prefill
                k = min(rng.randrange(1, 12), pl - c)
                sched.append({"a": (c, k)})
                c += k
            stop = rng.randrange(pl + 1, len(seq))
            sched += [{"a": (p, 1)} for p in range(pl, stop)]
            # preempted at `stop`: recomputed from scratch in chunks, then decode
            c = 0
            while c < stop:
                k = min(rng.randrange(1, 40), stop - c)
                sched.append({"a": (c, k)})
                c += k
            sched += [{"a": (p, 1)} for p in range(stop, len(seq))]
            got = run(make_controller(), {"a": seq}, {"a": pl}, sched)["a"]
            assert_same(self, got, want)

    def test_batched_requests(self):
        rng = random.Random(2)
        seqs, pls = {}, {}
        for r in "abcde":
            pls[r] = rng.randrange(3, 15)
            seqs[r] = random_sequence(rng, pls[r], rng.randrange(40, 100))
        pos = {r: 0 for r in seqs}
        sched = []
        while any(pos[r] < len(seqs[r]) for r in seqs):
            step = {}
            for r in seqs:
                if pos[r] >= len(seqs[r]) or rng.random() < 0.2:  # sometimes not scheduled
                    continue
                k = pls[r] - pos[r] if pos[r] < pls[r] else 1
                step[r] = (pos[r], k)
                pos[r] += k
            if step:
                sched.append(step)
        got = run(make_controller(), seqs, pls, sched)
        for r in seqs:
            assert_same(self, got[r], reference(seqs[r], pls[r]))

    def test_release_and_slot_reuse(self):
        ctl = make_controller()
        seq = [1, THINK, WAIT, N, NN]
        run(ctl, {"a": seq}, {"a": 2}, [{"a": (0, 2)}, {"a": (2, 1)}])
        ctl.release(["a"])
        self.assertNotIn("a", ctl._slot)
        got = run(ctl, {"b": seq}, {"b": 2},
                  [{"b": (0, 2)}] + [{"b": (p, 1)} for p in range(2, 5)])["b"]
        assert_same(self, got, reference(seq, 2))  # reused slot starts clean

    def test_prefix_cache_hit_starts_mid_prompt(self):
        rng = random.Random(3)
        pl = 20
        seq = random_sequence(rng, pl, 60)
        sched = [{"a": (12, pl - 12)}] + [{"a": (p, 1)} for p in range(pl, len(seq))]
        got = run(make_controller(), {"a": seq}, {"a": pl}, sched)["a"]
        assert_same(self, got, reference(seq, pl))


class TestSteerResidual(unittest.TestCase):
    def test_adds_coef_times_vector(self):
        m = types.SimpleNamespace(
            rkv_steer_coef=torch.tensor([0.0, -1.25, 0.0, 2.0, 7.0]),
            rkv_steer_vec=torch.tensor([1.0, 2.0, 3.0]),
        )
        h = torch.randn(4, 3, dtype=torch.bfloat16)
        res = torch.randn(4, 3, dtype=torch.bfloat16)
        h2, res2 = steering.steer_residual(m, h, res)
        self.assertTrue(torch.equal(h2, h))
        delta = (m.rkv_steer_coef[:4, None] * m.rkv_steer_vec).to(torch.bfloat16)
        self.assertTrue(torch.equal(res2, res + delta))
        self.assertTrue(torch.equal(res2[0], res[0]))  # coef 0 -> untouched
        h3, none = steering.steer_residual(m, h, None)  # before layer 0
        self.assertIsNone(none)
        self.assertTrue(torch.equal(h3, h + delta))


if __name__ == "__main__":
    unittest.main()
