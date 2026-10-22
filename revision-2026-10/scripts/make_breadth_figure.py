"""Breadth Evaluation figure and supplement rows (reconstructed during revision, 2026-10-12).
usage: python -I make_breadth_figure.py <out_pdf> <out_tex_dir> <tag>=<title>=<run_dir> ...
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
OUT, TEX = sys.argv[1:3]
RUNS = [a.split("=", 2) for a in sys.argv[3:]]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COL, INK, MUTED = "#eb6834", "#222222", "#777777"
order = [(m, k) for m in ["Logistic", "RandomForest", "XGBoost"] for k in (1, 100)]
lab = {"Logistic": "LR", "RandomForest": "RF", "XGBoost": "XGB"}
def load(d):
    if str(d).endswith(".csv"):
        x = pd.read_csv(d)
        return x.pivot_table(index=["model", "K", "repeat"], columns="engine", values="p99"), x.pivot_table(index=["model", "K", "repeat"], columns="engine", values="p50")
    x = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(Path(d) / "lat__*.parquet")))])
    g = x.groupby(["model", "K", "repeat", "engine"]).latency_ms
    return g.quantile(.99).unstack("engine"), g.quantile(.5).unstack("engine")
if len(RUNS) == 4:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2)); axes = axes.ravel()
else:
    fig, axes = plt.subplots(len(RUNS), 1, figsize=(3.4, 2.05 * len(RUNS)))
Path(TEX).mkdir(parents=True, exist_ok=True)
for ax, (tag, title, d) in zip(np.atleast_1d(axes), RUNS):
    p99, p50 = load(d)
    rng = np.random.default_rng(7)
    for i, g in enumerate(order):
        r = (p99.loc[g, "B1"] / p99.loc[g, "A"]).values
        cv = p99.loc[g, "B2"].std(ddof=1) / p99.loc[g, "B2"].mean()
        x = i + rng.uniform(-.12, .12, len(r))
        if cv > .25:
            ax.scatter(x, r, s=16, facecolor="white", edgecolor=COL, linewidth=1.0, zorder=3)
        else:
            ax.scatter(x, r, s=16, color=COL, edgecolor="white", linewidth=.6, zorder=3)
        ax.hlines(np.median(r), i - .25, i + .25, color=INK, linewidth=1.6, zorder=4)
    ax.axhline(1, color=MUTED, linewidth=.8); ax.axhline(1.1, color=MUTED, linewidth=.8, linestyle=":")
    ax.set_yscale("log"); ax.set_ylim(0.45, 5.2); ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{lab[m]}\nK={k}" for m, k in order], fontsize=7)
    ax.set_ylabel("P99(B1)/P99(A)"); ax.set_title(title, fontsize=8, loc="left")
    rows = []
    for (m, k, r), v in p99.iterrows():
        w = p50.loc[(m, k, r)]
        rows.append(f"{m.replace('RandomForest','Random forest')} & {k} & {r} & {w.A:.3f} & {w.B1:.3f} & {w.B2:.3f} & {v.A:.3f} & {v.B1:.3f} & {v.B2:.3f} & {v.B1/v.A:.3f} \\\\")
    (Path(TEX) / f"s_{tag}_repeats.tex").write_text("\n".join(rows))
    m50 = p50.groupby(level=[0, 1]).mean()
    (Path(TEX) / f"s_{tag}_p50.tex").write_text("\n".join(f"{m.replace('RandomForest','Random forest')} & {k} & {v.A:.3f} & {v.B1:.3f} & {v.B2:.3f} & {v.B1/v.A:.2f} \\\\" for (m, k), v in m50.iterrows()))
    print(tag, (p50.B1 / p50.A).min().round(3), (p50.B1 / p50.A).max().round(3), int(((p50.B1 / p50.A) > 1).sum()))
fig.tight_layout(); fig.savefig(OUT); print("fig ok")
