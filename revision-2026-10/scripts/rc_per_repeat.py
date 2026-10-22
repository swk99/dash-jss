"""Recompute client-CPU-restriction (RC) per-repeat values. usage: python -I rc_per_repeat.py <repo> <out.csv>"""
import sys, glob, pandas as pd
R, O = sys.argv[1:3]
d = pd.concat([pd.read_parquet(f).assign(cond=f.split('RC__')[1].split('__')[0]) for f in glob.glob(f"{R}/experiments/2026-10-02/robustness/ckpt/RC__*.parquet")])
assert d.success.all() and d.groupby(['cond','repeat','engine']).size().eq(5000).all()
g = d.groupby(['cond','repeat','engine']).latency_ms
q = g.quantile(.99).unstack('engine').add_prefix('p99_'); p = g.quantile(.5).unstack('engine').add_prefix('p50_')
o = q.join(p); o['ratio_p99_B1_A'] = o.p99_B1_RedisFetch / o.p99_A_ProposedStateful
o.to_csv(O); print(o.round(3))
