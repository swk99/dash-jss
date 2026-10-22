"""Read-only audit; usage: python reanalyse.py /path/to/repo /path/to/output"""
import ast,hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score,average_precision_score
R=Path(sys.argv[1]).resolve(); O=Path(sys.argv[2]).resolve();O.mkdir(parents=True,exist_ok=True)
E=R/'experiments/2026-10-02'; h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=json.loads((E/'replication/expected_sha256.json').read_text())
checks=[]
for section in ['artifacts','repo_py']:
 for name,value in expected[section].items():
  p=R/name;checks.append(dict(section=section,file=name,present=p.exists(),expected=value,actual=h(p) if p.exists() else None,match=p.exists() and h(p)==value))
pd.DataFrame(checks).to_csv(O/'hash_checks.csv',index=False)
raw=pd.concat([pd.read_parquet(p).assign(env=p.parent.name[5:],source=str(p.relative_to(R))) for p in sorted((E/'replication').glob('ckpt_*/lat__*.parquet'))],ignore_index=True)
assert len(raw)==300000 and raw.success.all() and np.isfinite(raw.latency_ms).all()
assert not raw.duplicated(['env','K','repeat','engine','request_sequence']).any()
rows=[]
for key,g in raw.groupby(['env','K','repeat','engine']):
 assert len(g)==5000
 rows.append(dict(zip(['env','K','repeat','engine'],key),n=len(g),p50=g.latency_ms.quantile(.5),p99=g.latency_ms.quantile(.99),mean=g.latency_ms.mean(),state_mean=g.state_ms.mean(),stage2_rate=g.stage2_fired.mean()))
s=pd.DataFrame(rows);s.to_csv(O/'replication_per_repeat.csv',index=False)
p=s.pivot(index=['env','K','repeat'],columns='engine',values='p99');p['ratio_B1_A']=p.B1_RedisFetch/p.A_ProposedStateful;p.to_csv(O/'replication_p99_ratios.csv')
summary=[]
for (env,k),g in p.groupby(level=['env','K']):
 summary.append(dict(env=env,K=k,median_ratio=g.ratio_B1_A.median(),above_one=int((g.ratio_B1_A>1).sum()),B2_cv=g.B2_InMemoryRecompute.std(ddof=1)/g.B2_InMemoryRecompute.mean(),ratios=g.ratio_B1_A.tolist()))
eq=[]
for f in sorted((E/'replication').glob('ckpt_*/e0__*.parquet')):
 d=pd.read_parquet(f);m=json.loads(f.with_suffix('.json').read_text());piv=d.pivot(index='tick_index',columns='engine',values='prediction')
 for eng,g in d.groupby('engine'):
  eq.append(dict(env=f.parent.name[5:],model=m['model'],engine=eng,n=len(g),mismatches=int((piv[eng]!=piv.A_ProposedStateful).sum()),fired=int(g.stage2_fired.sum()),label1=int(((g.prediction==1)&g.stage2_fired).sum()),label2=int(((g.prediction==2)&g.stage2_fired).sum()),window_start=m['window_start']))
pd.DataFrame(eq).to_csv(O/'equivalence.csv',index=False)
wraw=pd.concat([pd.read_parquet(f) for f in (E/'robustness/ckpt').glob('RW__*.parquet')]);ws=[]
for key,g in wraw.groupby(['W','repeat','engine']):
 ws.append(dict(zip(['W','repeat','engine'],key),n=len(g),state_mean=g.state_ms.mean(),p99=g.latency_ms.quantile(.99)))
ws=pd.DataFrame(ws);ws.to_csv(O/'window_per_repeat.csv',index=False)
wsummary=ws.groupby(['W','engine']).agg(state_mean=('state_mean','mean'),p99_mean=('p99','mean')).reset_index();wsummary.to_csv(O/'window_summary.csv',index=False)
d=pd.read_csv(E/'state_checks/dash_paper_results.csv',low_memory=False)
m=d[d.record_type=='memory_raw'];m.groupby(['W','N']).agg(mean_bytes=('redis_mem_bytes','mean'),sd_bytes=('redis_mem_bytes','std'),list_len=('redis_list_len','mean'),count=('redis_mem_bytes','size')).to_csv(O/'memory_summary.csv')
u=d[(d.experiment=='lua_update_cost')&(d.record_type=='latency_tick')];ur=[]
for key,g in u.groupby(['update_method','repeat']):
 ur.append(dict(method=key[0],repeat=key[1],n=len(g),p50_us=1000*g.latency_ms.quantile(.5),p99_us=1000*g.latency_ms.quantile(.99)))
us=pd.DataFrame(ur);us.to_csv(O/'update_per_repeat.csv',index=False);us.groupby('method').agg(p50_mean=('p50_us','mean'),p50_sd=('p50_us','std'),p99_mean=('p99_us','mean'),p99_sd=('p99_us','std')).to_csv(O/'update_summary.csv')
legacy=[]
for f in sorted((R/'results').glob('latency*.csv')):
 d=pd.read_csv(f)
 legacy.append(dict(file=str(f.relative_to(R)),rows=len(d),thresholds=sorted(d.tau.unique().tolist()) if 'tau' in d else [],trial_unique=d.trial.nunique() if 'trial' in d else None,columns=list(d.columns)))
# Execute only the exact threshold helper AST, avoiding training imports/side effects.
tree=ast.parse((R/'train_2.py').read_text(encoding='utf-8-sig')); names={'_safe_div','_f_beta','choose_threshold'}
ns=dict(np=np,precision_recall_curve=precision_recall_curve,average_precision_score=average_precision_score,Optional=__import__('typing').Optional,Dict=dict,Any=__import__('typing').Any)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'extracted_threshold_helpers','exec'),ns)
y=np.array([0,1,0,1,0,1]);prob=np.array([.1,.2,.3,.4,.5,.6]);v=ns['choose_threshold'](y,prob);pred=prob>=v['thr'];pr,rc,th=precision_recall_curve(y,prob);scores=2*pr[:-1]*rc[:-1]/(pr[:-1]+rc[:-1]+1e-12);best=int(np.argmax(scores))
bug=dict(test_kind='synthetic unit diagnostic, NOT empirical model recalibration',y=y.tolist(),probabilities=prob.tolist(),legacy_return=v,actual_at_returned_threshold=dict(precision=precision_score(y,pred),recall=recall_score(y,pred),f1=f1_score(y,pred)),correct_f1_max_threshold=float(th[best]),correct_f1_max=float(scores[best]))
assert abs(v['f1']-f1_score(y,pred))>1e-6
report=dict(commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),audit_type='independent reanalysis of existing records; no new latency measurements',replication_rows=len(raw),failures=int((~raw.success).sum()),replication_summary=summary,equivalence_rows=sum(x['n'] for x in eq),equivalence_mismatches=sum(x['mismatches'] for x in eq),window_rows=len(wraw),update_rows=len(u),memory_rows=len(m),hash_all_match=all(x['match'] for x in checks),legacy=legacy,threshold_diagnostic=bug,missing_input_files=len(expected['data_files']) if 'data_files' in expected else 'see manifest')
(O/'audit.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)));print(json.dumps({k:v for k,v in report.items() if k!='legacy'},indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
