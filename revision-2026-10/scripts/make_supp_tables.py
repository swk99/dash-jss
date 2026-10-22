"""Generate supplement LaTeX tables from recomputed records (revision v4).
usage: python -I make_supp_tables.py <reaudit_dir> <extension_run_dir> <repo> <out_dir>"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
RA, EXT, REPO, OUT = (Path(a) for a in sys.argv[1:5]); OUT.mkdir(parents=True, exist_ok=True)
f = lambda x, n=3: f"{x:.{n}f}"
# S4a original per-repeat
s = pd.read_csv(RA/'replication_per_repeat.csv')
p99 = s.pivot_table(index=['env','K','repeat'], columns='engine', values='p99')
p50 = s.pivot_table(index=['env','K','repeat'], columns='engine', values='p50')
rows=[]
for (e,k,r),v in p99.iterrows():
    a,b1,b2=v['A_ProposedStateful'],v['B1_RedisFetch'],v['B2_InMemoryRecompute']
    q=p50.loc[(e,k,r)]
    rows.append(f"{e.capitalize()} & {k} & {r} & {f(q['A_ProposedStateful'])} & {f(q['B1_RedisFetch'])} & {f(q['B2_InMemoryRecompute'])} & {f(a)} & {f(b1)} & {f(b2)} & {f(b1/a)} \\\\")
(OUT/'s_orig_repeats.tex').write_text("\n".join(rows))
# S4b extension per-repeat
d = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(EXT/'lat__*.parquet')))])
g = d.groupby(['model','K','repeat','engine']).latency_ms
q99 = g.quantile(.99).unstack('engine'); q50 = g.quantile(.5).unstack('engine')
rows=[]
for (m,k,r),v in q99.iterrows():
    w=q50.loc[(m,k,r)]
    rows.append(f"{m.replace('RandomForest','Random forest')} & {k} & {r} & {f(w.A)} & {f(w.B1)} & {f(w.B2)} & {f(v.A)} & {f(v.B1)} & {f(v.B2)} & {f(v.B1/v.A)} \\\\")
(OUT/'s_ext_repeats.tex').write_text("\n".join(rows))
# P50 means extension
m50 = q50.groupby(level=[0,1]).mean()
rows=[f"{m.replace('RandomForest','Random forest')} & {k} & {f(v.A)} & {f(v.B1)} & {f(v.B2)} & {f(v.B1/v.A,2)} \\\\" for (m,k),v in m50.iterrows()]
(OUT/'s_ext_p50.tex').write_text("\n".join(rows))
# S3 update per repeat
u = pd.read_csv(RA/'update_per_repeat.csv')
rows=[f"{r.method.replace('pipeline_nontx','pipeline')} & {int(r.repeat)} & {r.p50_us:.1f} & {r.p99_us:.1f} \\\\" for r in u.sort_values(['method','repeat']).itertuples()]
(OUT/'s_update_repeats.tex').write_text("\n".join(rows))
# S5 window per repeat ratios
w = pd.read_csv(RA/'window_per_repeat.csv'); pv=w.pivot_table(index=['W','repeat'],columns='engine',values='p99')
rr=(pv['B1_RedisFetch']/pv['A_ProposedStateful']).unstack('repeat')
rows=[f"{W} & "+" & ".join(f(x) for x in rr.loc[W].values)+" \\\\" for W in rr.index]
(OUT/'s_window_ratios.tex').write_text("\n".join(rows))
# S7 hashes
exp = json.loads((REPO/'experiments/2026-10-02/replication/expected_sha256.json').read_text())
rows=[f"\\texttt{{{k.replace('_','\\_')}}} & \\texttt{{{v[:16]}\\ldots}} \\\\" for k,v in exp['artifacts'].items()]
(OUT/'s_hashes.tex').write_text("\n".join(rows))
print({k:(v if isinstance(v,str) else '...') for k,v in exp.items()})
print(open(OUT/'s_ext_p50.tex').read())
