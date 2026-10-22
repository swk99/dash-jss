"""Prepared extension, NOT executed on research hosts in this audit.
Keep this script outside the frozen repo. See experiment_protocol.txt.
Preflight only needs the standard library. --run requires the original runtime.
"""
import argparse,glob,hashlib,importlib.util,json,os,platform,re,subprocess,sys,uuid
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--repo',required=True);a.add_argument('--data-glob',required=True);a.add_argument('--env-label',required=True);a.add_argument('--out',required=True);a.add_argument('--models',nargs='+',choices=['Logistic','RandomForest','XGBoost'],default=['Logistic','RandomForest','XGBoost']);a.add_argument('--new-data',action='store_true',help='Explicitly declare a separate dataset extension, never an original-input replication');a.add_argument('--run',action='store_true');a=a.parse_args()
 repo=Path(a.repo).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 script=repo/'experiments/2026-10-02/replication/run_replication.py';expected=json.loads(script.with_name('expected_sha256.json').read_text())
 files=glob.glob(a.data_glob);issues=[]
 actual={Path(f).name:sha(f) for f in files}
 if not files:issues.append('NO INPUT FILES')
 if actual!=expected['data_files'] and not a.new_data:issues.append('INPUT FILE SET/HASH MISMATCH: need all original 75 flush parquet files')
 for section in ['artifacts','repo_py']:
  for n,h in expected[section].items():
   if not (repo/n).is_file() or sha(repo/n)!=h:issues.append(section+':'+n)
 if sha(script)!=expected['script_sha256']:issues.append('original replication driver hash')
 report={'status':'BLOCKED' if issues else 'PREFLIGHT_PASSED','issues':issues,'input_count':len(files),'new_latency_measurements':False,'time_utc':datetime.now(timezone.utc).isoformat()}
 (out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if issues:return 2
 if not a.run:return 0
 # New run-specific directory; no resuming or overwriting historical measurements.
 run=out/('extension_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);run.mkdir()
 for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[v]='1'
 os.environ.update(DASH_REPO=str(repo),DASH_DATA_GLOB=a.data_glob,DASH_ENV_LABEL=a.env_label,DASH_OUT=str(run),DASH_ALLOW_MISMATCH='1' if a.new_data else '0')
 spec=importlib.util.spec_from_file_location('frozen_replication',script);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
 # Original driver verifies all hashes; only declared input differences may pass.
 if a.new_data and R.SHA_CHECK not in ['all_match','MISMATCH:input_array_sha256,data_file_manifest_sha256']:raise RuntimeError('Unexpected mismatch: '+R.SHA_CHECK)
 import numpy as np,pandas as pd,importlib.metadata as md
 r=R.client();r.ping();R.PFX='dash_extension:'+uuid.uuid4().hex+':'
 rules=dict(R.PREREG,extension='apply unchanged rules separately for each model and K; do not pool hosts/models',model_order='seeded permutation of requested models per K/repeat; same engine order seed as original',new_study='new measurement occasion, not same physical hosts unless operator verifies')
 (run/'rules_before_measurement.json').write_text(json.dumps(rules,indent=2))
 meta=dict(env_label=a.env_label,python=platform.python_version(),platform=platform.platform(),cpu_count=os.cpu_count(),redis_server=r.info('server'),redis_persistence=r.info('persistence'),versions={p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','redis','torch','pyarrow']},extension_script_sha256=sha(__file__),original_script_sha256=sha(script),input_sha256=R.INPUT_SHA,manifest_sha256=R.MANIFEST_SHA,artifacts=R.ART,dataset_status='new dataset extension' if a.new_data else 'original input',sha_check=R.SHA_CHECK,repo_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),models=a.models,run_status='started',measured_cells=0)
 (run/'metadata.json').write_text(json.dumps(meta,indent=2,default=str));summ=[];cpu=[]
 try:
  for K in R.KS:
   offs,L,rr=R.schedule(K)
   for rep in range(R.REPS):
    model_order=list(np.random.default_rng([R.SEED,K,rep,20261011]).permutation(a.models))
    for model in model_order:
     h,threads,stdout,warnings=R.load(model);meta.setdefault('model_load',{})[model]=dict(threads=threads,stdout=stdout,warnings=warnings)
     if warnings:raise RuntimeError('Model version warning: resolve compatibility before publication run')
     order=list(np.random.default_rng([R.SEED,rep,K,0,4]).permutation(['A','B1','B2']));parts=[]
     for eng in order:
      mon=R.CpuMon()
      try:rows,keys=R.run_engine(eng,h,r,K,offs,L,rr,f'{model}:K{K}:r{rep}')
      finally:cv=mon.close()
      if keys:r.delete(*keys)
      d=pd.DataFrame(rows,columns=R.RCOLS);d['engine']=eng;d['model']=model;d['K']=K;d['repeat']=rep;d['env']=a.env_label;d['engine_order']='>'.join(order);d['model_order']='>'.join(model_order);d['tau_s1']=h.tau_conf;d['tau_s2']=h.tau_s2;parts.append(d)
      cpu.append(dict(model=model,K=K,repeat=rep,engine=eng,host_cpu_mean=float(np.mean(cv)) if cv else None))
     d=pd.concat(parts,ignore_index=True);d.to_parquet(run/f'lat__{model}__K{K}__r{rep}.parquet',index=False)
     if not d.success.all():raise RuntimeError('Failed requests retained; run is incomplete')
     piv=d.pivot(index='request_sequence',columns='engine',values='prediction')
     fire=d.pivot(index='request_sequence',columns='engine',values='stage2_fired')
     if not (piv.eq(piv.A,axis=0).all().all() and fire.eq(fire.A,axis=0).all().all()):raise RuntimeError('Prediction/gate mismatch; do not publish latency as equivalent workloads')
     for eng,g in d.groupby('engine'):
      summ.append(dict(env=a.env_label,model=model,K=K,repeat=rep,engine=eng,n=len(g),p50=g.latency_ms.quantile(.5),p99=g.latency_ms.quantile(.99),mean=g.latency_ms.mean(),state_mean=g.state_ms.mean(),stage2_rate=g.stage2_fired.mean()))
     meta['measured_cells']+=1;print(model,K,rep,'complete',flush=True)
  pd.DataFrame(summ).to_csv(run/'per_repeat.csv',index=False);pd.DataFrame(cpu).to_csv(run/'cpu.csv',index=False);meta['run_status']='complete'
 except Exception as e:
  meta['run_status']='failed';meta['error']=repr(e);raise
 finally:
  keys=list(r.scan_iter('*'+R.PFX+'*'))
  if keys:r.delete(*keys)
  meta['end_utc']=datetime.now(timezone.utc).isoformat();(run/'metadata.json').write_text(json.dumps(meta,indent=2,default=str));R.LOG.close()
 return 0
if __name__=='__main__':sys.exit(main())
