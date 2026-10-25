#!/usr/bin/env bash
# Re-render the three README figures from the result JSONs and copy them to
# benchmark/figures/ (series without results are simply absent).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
(cd math && $P plot_acc_tput.py >/dev/null && cp acc_vs_tput.png ../../figures/skipkv_acc_vs_tput.png)
(cd math_steer && $P plot_strength.py >/dev/null && cp steer_strength.png ../../figures/skipkv_steer_strength.png)
(cd math_stats && $P plot_preempt_tput.py >/dev/null && cp preempt_vs_tput.png ../../figures/skipkv_preempt_vs_tput.png)
echo "figures updated"
