"""Summarise a completed new model run. Never combines environments/datasets."""
from pathlib import Path
import sys,json,numpy as np,pandas as pd
P=Path(sys.argv[1]); meta=json.loads((P/'metadata.json').read_text());assert meta['run_status']=='complete'
s=pd.read_csv(P/'per_repeat.csv');rows=[]; raw_count=0;mismatches=0;failures=0
for f in P.glob('lat__*.parquet'):
 d=pd.read_parquet(f);raw_count+=len(d);failures+=int((~d.success).sum());assert not d.duplicated(['engine','request_sequence']).any()
 p=d.pivot(index='request_sequence',columns='engine',values='prediction');mismatches+=int((p[['B1','B2']].ne(p.A,axis=0)).sum().sum())
for (env,model,k),g in s.groupby(['env','model','K']):
 p=g.pivot(index='repeat',columns='engine',values='p99');q=p.B1/p.A;cv=p.B2.std(ddof=1)/p.B2.mean()
 row=dict(env=env,model=model,K=int(k),repeats=len(p),median_ratio_B1_A=float(q.median()),repeats_above_one=int((q>1).sum()),B2_cv=float(cv),noise_flag=bool(cv>.25),scope_limited=bool(q.median()<=1.1),ratios=q.tolist())
 for eng,e in g.groupby('engine'):
  for col in ['p50','p99','mean','state_mean','stage2_rate']:row[eng+'_'+col+'_mean']=float(e[col].mean())
  row[eng+'_p99_sd']=float(e.p99.std(ddof=1))
 rows.append(row)
pd.DataFrame(rows).drop(columns='ratios').to_csv(P/'summary.csv',index=False)
r=dict(status='complete',raw_rows=raw_count,failures=failures,prediction_mismatches_B1_B2_vs_A=mismatches,scope='new November dataset, one new shared Linux audit host; not original two-environment replication',host_interference='Document preparation continued during measurement. A brief independent plotting/host-inspection task ran near final cells; no observations were removed. Treat as exploratory extension, not isolated-host confirmation.',conditions=rows)
(P/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
