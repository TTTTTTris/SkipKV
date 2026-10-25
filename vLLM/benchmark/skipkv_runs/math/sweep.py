"""Run the MATH-500 sweep: one eval.py job per free GPU, skip jobs with a result.

    GPUS=0,1 python sweep.py [label ...]     # default: every GPU nvidia-smi lists

Run it with the Python environment that has the patched vLLM (or set PYTHON). Hugging Face
settings such as HF_HOME / HF_HUB_OFFLINE are passed through from the environment.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# RUN_DIR: where results/logs go (default: next to this script);
# EXTRA_ARGS: extra eval.py flags for every job (e.g. "--stats");
# PYTHON: interpreter for the eval.py jobs (default: the one running this script).
RUN_DIR = os.path.abspath(os.environ.get("RUN_DIR", HERE))
EXTRA = os.environ.get("EXTRA_ARGS", "").split()
os.makedirs(RUN_DIR, exist_ok=True)
EVAL = os.path.join(HERE, "..", "..", "eval.py")
PY = os.environ.get("PYTHON", sys.executable)


def all_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [int(i) for i in out.split()]


GPUS = [int(g) for g in os.environ["GPUS"].split(",")] if os.environ.get("GPUS") else all_gpus()


def idle_gpus():
    """GPUs using < 2 GB (so jobs started by an earlier scheduler, or by others, are left alone)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(i) for i, m in (l.split(",") for l in out.strip().splitlines()) if int(m) < 2048}

METHODS = {
    "fullkv": {},
    "rkv": {},
    "rkv_head": {"VLLM_V1_SKIPKV_GRANULARITY": "head"},
    "skipkv": {"VLLM_V1_SKIPKV_SENTENCE_THRESHOLD": "0.95"},
    "skipkv_head": {"VLLM_V1_SKIPKV_SENTENCE_THRESHOLD": "0.95", "VLLM_V1_SKIPKV_GRANULARITY": "head"},
}
# Activation steering (reference R1-Qwen-7B settings: layer 20, gamma 0.02). "_steer" uses
# the reference coefficient -1.25; "_steer_c1" uses -1.0.
STEER_VEC = os.path.join(HERE, "..", "..", "..", "..", "steering_vectors",
                         "DeepSeek-R1-Distill-Qwen-7B", "layer_20_transition_reflection_steervec.pt")
for base in ("skipkv", "skipkv_head"):
    for suffix, coef in (("_steer", "-1.25"), ("_steer_c1", "-1.0")):
        METHODS[base + suffix] = dict(METHODS[base], VLLM_V1_SKIPKV_STEER_VECTOR=STEER_VEC,
                                      VLLM_V1_SKIPKV_STEER_COEF=coef)
TASKS = {
    "math": (["--n", "500"], [512, 1024, 2048]),
}
COMMON = ["--max-tokens", "32768", "--max-model-len", "34816", "--mem-frac", "0.85"]
BASE_ENV = {
    "RKV_MODEL": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "VLLM_V1_R_KV_FREE_BLOCKS": "1",
    "VLLM_V1_R_KV_SCORE_CHUNK_MB": "2048",
}

jobs = []
for task, (targs, budgets) in TASKS.items():
    for m, menv in METHODS.items():
        for b in [None] if m == "fullkv" else budgets:
            label = f"{task}_{m}" + (f"_b{b}" if b else "")
            env = dict(BASE_ENV, **menv)
            if b:
                env.update(VLLM_V1_R_KV_BUDGET=str(b), VLLM_V1_R_KV_BUFFER="128")
            args = targs + COMMON + EXTRA + ["--label", label,
                                             "--out", os.path.join(RUN_DIR, label + ".json"),
                                             "--save-outputs", os.path.join(RUN_DIR, label + ".jsonl")]
            if m == "fullkv":
                args.append("--no-prefix")
            jobs.append((label, ["--task", task] + args, env))
only = sys.argv[1:]
# A job with a .log was already started (possibly by an earlier scheduler instance).
jobs = [j for j in jobs if (not only or j[0] in only) and not os.path.exists(os.path.join(RUN_DIR, j[0] + ".log"))]


def prev_wall(label):  # longest jobs first, using an earlier run's wall time if any
    f = os.path.join(HERE, label + ".json")
    return json.load(open(f))["wall_s"] if os.path.exists(f) else 0.0


jobs.sort(key=lambda j: -prev_wall(j[0]))
print(f"{len(jobs)} jobs on GPUs {GPUS}", flush=True)

running = {}
while jobs or running:
    for g, (label, p) in list(running.items()):
        if p.poll() is not None:
            print(f"{time.strftime('%H:%M:%S')} done  {label} (gpu {g}, rc={p.returncode})", flush=True)
            del running[g]
    idle = idle_gpus() if jobs else set()
    for g in GPUS:
        if g not in running and g in idle and jobs:
            label, args, env = jobs.pop(0)
            log = open(os.path.join(RUN_DIR, label + ".log"), "w")
            e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), **env)
            running[g] = (label, subprocess.Popen([PY, EVAL] + args, env=e, stdout=log, stderr=subprocess.STDOUT))
            print(f"{time.strftime('%H:%M:%S')} start {label} (gpu {g})", flush=True)
    time.sleep(30)
