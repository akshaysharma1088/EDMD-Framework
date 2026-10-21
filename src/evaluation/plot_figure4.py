# Draws Figure 4 of the paper from artifacts/simulation/main_runs.json
# Usage: python src/evaluation/plot_figure4.py artifacts/simulation/main_runs.json "Figure 4.png"
import json, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); out = sys.argv[2]
archs = ["EDMD", "Feature store", "Kappa", "Three-tier Medallion", "Lambda"]
runs = d["runs"]
m = lambda a, k: np.array([r["arch"][a][k] for r in runs])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
# (a) AUC
labels = ["Oracle (complete)"] + archs
means = [np.mean([r["oracle_auc"] for r in runs])] + [m(a, "auc").mean() for a in archs]
sds = [np.std([r["oracle_auc"] for r in runs], ddof=1)] + [m(a, "auc").std(ddof=1) for a in archs]
y = np.arange(len(labels))[::-1]
cols = ["#888888", "#4B2E83"] + ["#2F6DB5"] * 4
for yi, mu, sd, c in zip(y, means, sds, cols):
    ax1.errorbar(mu, yi, xerr=sd, fmt="o", color=c, ms=7, capsize=4, lw=1.5)
    ax1.text(mu + sd + 0.0008, yi, f"{mu:.3f}", va="center", fontsize=10)
ax1.set_yticks(y); ax1.set_yticklabels(labels)
ax1.set_xlabel("Serving-time AUC (mean ± SD, 5 seeds)")
ax1.set_xlim(0.840, 0.872)
ax1.set_title("(a) Model quality", loc="left", fontsize=11)
for s in ["top", "right"]: ax1.spines[s].set_visible(False)
# (b) TSS vs LERR
mk = {"EDMD": "D", "Feature store": "s", "Kappa": "^", "Three-tier Medallion": "v", "Lambda": "o"}
off = {"EDMD": (8, -14), "Feature store": (-8, -16), "Kappa": (8, 4), "Three-tier Medallion": (8, 4), "Lambda": (6, 8)}
for a in archs:
    x, yv = m(a, "tss").mean(), m(a, "lerr").mean()
    ax2.scatter(x, yv, marker=mk[a], s=70, color="#4B2E83" if a == "EDMD" else "#2F6DB5", zorder=3)
    ax2.annotate(a, (x, yv), textcoords="offset points", xytext=off[a], fontsize=10,
                 ha="right" if off[a][0] < 0 else "left")
ax2.set_xlabel("Training-serving skew, TSS (lower is better)")
ax2.set_ylabel("Late-event recovery, LERR")
ax2.set_xlim(-0.004, 0.075); ax2.set_ylim(0.4, 1.05)
ax2.set_title("(b) Consistency and completeness", loc="left", fontsize=11)
for s in ["top", "right"]: ax2.spines[s].set_visible(False)
ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out, dpi=300)
