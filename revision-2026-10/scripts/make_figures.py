"""Regenerate DASH v4 figures from archived raw-derived CSVs.

Reconstructed during the v4 revision (2026-10-11). These are NOT recovered
original plotting scripts. Inputs are the per-repeat CSVs produced by
reanalyse.py from the pinned repository (53353760) and the raw parquet records
of the single-host model extension.

usage: python -I make_figures.py <reaudit_dir> <extension_run_dir> <out_dir>
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RA, EXT, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
C = {"A": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a"}
MK = {"A": "o", "B1": "s", "B2": "^"}
INK, MUTED = "#222222", "#777777"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
ENG = {"A_ProposedStateful": "A", "B1_RedisFetch": "B1", "B2_InMemoryRecompute": "B2"}


def strip(ax, groups, ratios_by_group, color, hollow=()):
    rng = np.random.default_rng(7)  # fixed jitter, display only
    for i, g in enumerate(groups):
        r = np.asarray(ratios_by_group[g])
        x = i + rng.uniform(-0.12, 0.12, len(r))
        if g in hollow:
            ax.scatter(x, r, s=16, facecolor="white", edgecolor=color, linewidth=1.0, zorder=3)
        else:
            ax.scatter(x, r, s=16, color=color, zorder=3, edgecolor="white", linewidth=0.6)
        ax.hlines(np.median(r), i - 0.25, i + 0.25, color=INK, linewidth=1.6, zorder=4)
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    ax.axhline(1.10, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(groups)))


# 1. original two-environment ratios
s = pd.read_csv(RA / "replication_per_repeat.csv")
pv = s.pivot_table(index=["env", "K", "repeat"], columns="engine", values="p99")
pv["r"] = pv["B1_RedisFetch"] / pv["A_ProposedStateful"]
groups = [("cloud", 1), ("cloud", 100), ("laptop", 1), ("laptop", 100)]
rb = {g: pv.loc[g, "r"].values for g in groups}
fig, ax = plt.subplots(figsize=(3.4, 2.2))
strip(ax, groups, rb, C["B1"])
ax.set_xticklabels([f"{e.capitalize()}\nK={k}" for e, k in groups])
ax.set_ylabel("P99(B1) / P99(A), per repeat")
fig.tight_layout(); fig.savefig(OUT / "fig_ratio_two_env.pdf"); plt.close(fig)

# 2. state interval vs W
w = pd.read_csv(RA / "window_per_repeat.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for eng, lab in ENG.items():
    g = w[w.engine == eng].groupby("W").state_mean.agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], color=C[lab], marker=MK[lab], ms=4,
                linewidth=1.5, capsize=2, label=lab)
    ax.annotate(lab, (g.index[-1], g["mean"].iloc[-1]), xytext=(5, 0), textcoords="offset points",
                va="center", color=INK, fontsize=8)
ax.set_xlabel("Window length W (rows)"); ax.set_ylabel("Mean state interval (ms)")
ax.set_xticks([10, 30, 60, 120]); ax.set_xlim(0, 135); ax.set_ylim(0, None)
ax.legend(frameon=False, loc="upper left", ncol=3)
fig.tight_layout(); fig.savefig(OUT / "fig_state_vs_W.pdf"); plt.close(fig)

# 3. key memory
m = pd.read_csv(RA / "memory_summary.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.2))
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
for c, (W, g) in zip(cols, m.groupby("W")):
    a1.plot(g.N, g.mean_bytes, marker="o", ms=3.5, color=c, linewidth=1.5, label=f"W={int(W)}")
a1.set_xscale("log"); a1.set_xlabel("Rows written N"); a1.set_ylabel("Redis MEMORY USAGE (bytes)")
a1.set_ylim(0, 4700); a1.legend(frameon=False, fontsize=7, ncol=4, loc="upper left"); a1.set_title("(a) fixed W, varying N", fontsize=8, loc="left")
Ws = sorted(m.W.unique()); full = m[m.N == 20000].set_index("W").mean_bytes
a2.plot(Ws, [full[x] for x in Ws], marker="o", color=cols[0], linewidth=1.5, label="Measured key (N=20,000)")
a2.plot(Ws, [4 * 6 * x for x in Ws], marker="s", color=cols[1], linewidth=1.5, linestyle="--",
        label="Logical payload 4dW")
a2.set_xlabel("Window length W (rows)"); a2.set_ylabel("Bytes"); a2.set_xticks(Ws)
a2.legend(frameon=False, fontsize=7); a2.set_title("(b) varying W", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_memory.pdf"); plt.close(fig)

# 4. update-only cost (supplement)
u = pd.read_csv(RA / "update_per_repeat.csv")
order = [("sequential", "Sequential"), ("pipeline_nontx", "Pipeline"), ("lua", "Lua script")]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.0, 2.2))
for ax, col, ttl in [(b1, "p50_us", "(a) per-repeat P50"), (b2, "p99_us", "(b) per-repeat P99")]:
    for i, (k, lab) in enumerate(order):
        v = u[u.method == k][col].values
        ax.scatter(i + np.linspace(-0.1, 0.1, len(v)), v, s=16, color=C["A"], edgecolor="white", zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, linewidth=1.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in order])
    ax.set_ylabel("Update-only latency (µs)"); ax.set_title(ttl, fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "figS_update_cost.pdf"); plt.close(fig)

# 5. extension ratios (supplement)
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(EXT / "lat__*.parquet")))])
q = d.groupby(["model", "K", "repeat", "engine"]).latency_ms.quantile(0.99).unstack("engine")
q["r"] = q["B1"] / q["A"]
eg = [(mo, k) for mo in ["Logistic", "RandomForest", "XGBoost"] for k in (1, 100)]
rb = {g: q.loc[g, "r"].values for g in eg}
fig, ax = plt.subplots(figsize=(3.4, 2.3))
strip(ax, eg, rb, C["B1"], hollow={("RandomForest", 1)})  # B2 CV > 0.25: noisy
ax.set_xticklabels([f"{mo.replace('RandomForest','RF').replace('Logistic','LR').replace('XGBoost','XGB')}\nK={k}" for mo, k in eg], fontsize=7)
ax.set_ylabel("P99(B1) / P99(A), per repeat")
fig.tight_layout(); fig.savefig(OUT / "fig_ratio_extension.pdf"); plt.close(fig)
print("written", sorted(p.name for p in OUT.glob("*.pdf")))
