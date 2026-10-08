"""Forty fresh sequential W1 GPU fits; never fits W0 or selects by partial scores."""
import sys,os,json,subprocess,time,argparse,datetime,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'src'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def worker(local,index):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from dataclasses import replace
    from loster2026.config import Config,PILOT_DATASETS
    from loster2026.data import load_ucr
    from loster2026.training import fit
    from loster2026.checkpointing import trusted_load
    from loster2026.reproducibility import environment,source_hash
    out=local/'warp-validation';frozen=json.loads((out/'execution/frozen.json').read_text())
    assert source_hash()==frozen['source_sha256']
    assert environment()==frozen['environment']
    for r in frozen['source_files']:assert digest(REPO/r['path'])==r['sha256']
    name=PILOT_DATASETS[index//5];seed=index%5
    config=replace(Config(),dataset=name,seed=seed,device='cuda',warp_mode='MonotoneWarp',augmentation_order=('sign','equal-segment-permutation','monotone-time-warp'))
    verified=json.loads((local/'data/verification.json').read_text())['datasets'][name]
    data=load_ucr(local/'data/ucr',name,verified)
    assert data.metadata['input_sha256']==verified['input_sha256']
    result=fit(data,config,out)
    key=Path(name)/config.variant/('seed-%d'%seed)
    manifest_path=out/'artifacts'/key/'manifest.json';m=json.loads(manifest_path.read_text())
    assert m['checkpoint_round_trip_exact'] and m['status']=='completed'
    assert m['source_sha256']==frozen['source_sha256'] and m['environment']==frozen['environment']
    assert m['git']['sha']==frozen['git_sha']
    old=json.loads((local/'pilot/artifacts'/key/'manifest.json').read_text())
    new_init=trusted_load(out/'artifacts/initialization'/name/('seed-%d'%seed)/(m['paired_cache_key']+'.pt'),'cpu')
    old_init=trusted_load(local/'pilot/artifacts/initialization'/name/('seed-%d'%seed)/(old['paired_cache_key']+'.pt'),'cpu')
    for k,v in old_init['model_state'].items():
        if k.startswith('original.') or k=='centers_original':assert torch.equal(v,new_init['model_state'][k]),'Original preparation pairing lost: '+k
    assert old_init['pretraining_losses']['original']==new_init['pretraining_losses']['original']
    a=old_init['rng_joint_boundary'];b=new_init['rng_joint_boundary']
    assert a['python']==b['python'] and torch.equal(a['torch'],b['torch'])
    assert a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
    assert all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))
    for snap in ('A_train','A_init'):
        expected=np.load(out/'data-only'/name/('seed-%d'%seed)/(snap+'.npz'))['monotone']
        assert np.array_equal(new_init[snap],expected),'Data-only candidate changed'
        assert m['augmentation'][snap]['minimum_coordinate_difference']>0
    payload=trusted_load(manifest_path.parent/'complete.pt','cpu')
    record=out/'runs'/key;record.mkdir(parents=True,exist_ok=False)
    np.save(record/'predictions.npy',np.asarray(payload['assignments']),allow_pickle=False)
    from loster2026.metrics import contingency
    (record/'contingency.json').write_text(json.dumps(contingency(data.labels,np.asarray(payload['assignments'])).tolist()))
    (record/'validation.json').write_text(json.dumps(dict(status='COMPLETE',checkpoint_sha256=digest(manifest_path.parent/'complete.pt'),original_preparation_exact=True,rng_boundary_exact=True,data_only_snapshots_exact=True,checkpoint_round_trip_exact=True,source_sha256=source_hash()),indent=2))
    print('COMPLETE',index,name,seed,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--local-root',type=Path,required=True);p.add_argument('--worker-index',type=int);args=p.parse_args();local=args.local_root
    if args.worker_index is not None:worker(local,args.worker_index);return
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from loster2026.reproducibility import environment,source_hash
    out=local/'warp-validation';execution=out/'execution';execution.mkdir(exist_ok=False)
    source_files=[]
    for p in sorted(ROOT.rglob('*.py')):
        rel=p.relative_to(REPO);source_files.append(dict(path=str(rel),sha256=digest(p)))
    archive=execution/'source';archive.mkdir()
    import shutil
    for r in source_files:
        dest=archive/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/r['path'],dest)
    for p in [REPO/'analysis/experiments/07-warp-validity-resolution.md',REPO/'analysis/experiments/warp-validation-protocol.json']:
        shutil.copyfile(p,execution/p.name)
    frozen=dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_sha256=source_hash(),source_files=source_files,environment=environment(),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (execution/'frozen.json').write_text(json.dumps(frozen,indent=2))
    inventory=[dict(path=str(p.relative_to(local/'pilot')),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=digest(p)) for p in sorted((local/'pilot').rglob('*')) if p.is_file()]
    (execution/'pilot-before.json').write_text(json.dumps(inventory,indent=2))
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
    runs=[]
    for i in range(40):
        start=time.time()
        with (execution/('run-%02d-stdout.log'%i)).open('x') as stdout,(execution/('run-%02d-stderr.log'%i)).open('x') as stderr:
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--local-root',str(local),'--worker-index',str(i)],cwd=REPO,env=env,stdout=stdout,stderr=stderr)
        runs.append(dict(index=i,status='COMPLETE' if r.returncode==0 else 'FAILED',returncode=r.returncode,elapsed_seconds=time.time()-start))
        (execution/'ledger.json').write_text(json.dumps(runs,indent=2))
        print(json.dumps(dict(terminal=len(runs),complete=sum(r['status']=='COMPLETE' for r in runs),failed=sum(r['status']=='FAILED' for r in runs))),flush=True)
    for r in inventory:
        p=local/'pilot'/r['path'];assert p.stat().st_size==r['size'] and p.stat().st_mtime_ns==r['mtime_ns'] and digest(p)==r['sha256'],'Pilot changed: '+r['path']
    assert {str(p.relative_to(local/'pilot')) for p in (local/'pilot').rglob('*') if p.is_file()}=={r['path'] for r in inventory}
    (execution/'pilot-preservation.json').write_text(json.dumps(dict(files=len(inventory),all_bytes_and_mtimes_unchanged=True,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    print('VALIDATION_TERMINAL',flush=True)
if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support();main()
