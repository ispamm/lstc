"""Exact data-only historical replay; label-free candidate and diagnostics."""
import sys,json,csv,argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.interpolate import CubicSpline
from loster2026 import augmentation as legacy
from loster2026.config import Config,PILOT_DATASETS,PILOT_SEEDS
from loster2026.data import load_ucr
from loster2026.reproducibility import sha256_file,environment,source_hash

def candidate_coordinates(m,length):
    z=CubicSpline(np.linspace(0,length-1,len(m)),m-1)(np.arange(length))
    v=np.exp(z-z.max())
    c=np.r_[0.,np.cumsum((v[:-1]+v[1:])/2)]
    xp=c*((length-1)/c[-1]);xp[-1]=length-1
    if not np.isfinite(xp).all() or not np.all(np.diff(xp)>0):raise ValueError('Unrepresentable path; no rescue')
    return xp

def longest(mask):
    e=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return int((e[1::2]-e[::2]).max()) if len(e) else 0

def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None

def signal(x,threshold):
    x=x.astype(float);d=np.diff(x)
    return dict(finite_rate=float(np.isfinite(x).mean()),variance=float(x.var()),difference_energy=float(np.mean(d*d)),total_variation=float(np.abs(d).sum()),repeated_fraction=float((d==0).mean()),large_jump_fraction=float((np.abs(d)>threshold).mean()),longest_flat_intervals=longest(d==0))

def main():
    p=argparse.ArgumentParser();p.add_argument('--local-root',type=Path,required=True);args=p.parse_args()
    local=args.local_root;out=local/'warp-validation/data-only';out.mkdir(parents=True,exist_ok=False)
    expected=list(csv.DictReader((ROOT.parents[1]/'analysis/experiments/pilot/warp-diagnostics.csv').open()))
    summaries=[];records=[];references=[]
    interp0=np.interp;normal0=np.random.normal;spline0=legacy.CubicSpline
    initial_source=source_hash();env=environment()
    for name in PILOT_DATASETS:
        data=load_ucr(local/'data/ucr',name);n,length=data.series.shape
        for seed in PILOT_SEEDS:
            key=Path(name)/'LoSTer-Legacy-Clean'/('seed-%d'%seed)
            mp=local/'pilot/artifacts'/key/'manifest.json';saved=json.loads(mp.read_text())
            assert saved['environment']==env and saved['source_sha256']==initial_source
            assert saved['dataset']==data.metadata and saved['status']=='completed' and saved['checkpoint_round_trip_exact']
            from dataclasses import replace
            config=replace(Config(),dataset=name,seed=seed,device='cuda')
            assert json.dumps(saved['config'],sort_keys=True)==json.dumps(config.to_dict(),sort_keys=True)
            references.append(dict(dataset=name,seed=seed,manifest_sha256=sha256_file(mp),checkpoint_sha256=sha256_file(mp.parent/'complete.pt'),result_sha256=sha256_file(local/'pilot/results'/key/'metrics.json'),initialization_sha256=sha256_file(local/'pilot/artifacts/initialization'/name/('seed-%d'%seed)/(saved['paired_cache_key']+'.pt'))))
            np.random.seed(seed)
            for snapshot in ('A_train','A_init'):
                captured=[];draws=[];raw=[]
                def normal(*a,**kw):
                    r=normal0(*a,**kw);draws.append(r.copy());return r
                def interp(t,xp,fp,*a,**kw):
                    captured.append((xp.copy(),fp.copy()));return interp0(t,xp,fp,*a,**kw)
                def spline(*a,**kw):
                    fn=spline0(*a,**kw)
                    def call(t):
                        r=fn(t);raw.append(r.copy());return r
                    return call
                np.random.normal=normal;np.interp=interp;legacy.CubicSpline=spline
                try:h,info=legacy.augment(data.series,config)
                finally:np.random.normal=normal0;np.interp=interp0;legacy.CubicSpline=spline0
                row=next(r for r in expected if r['dataset']==name and int(r['seed'])==seed and r['snapshot']==snapshot)
                assert info['snapshot_sha256']==row['snapshot_sha256']==saved['augmentation'][snapshot]['snapshot_sha256']
                dest=out/name/('seed-%d'%seed);dest.mkdir(parents=True,exist_ok=True)
                hx=np.stack([c[0] for c in captured]);mx=np.empty_like(hx)
                before=np.stack([r*((length-1)/r[-1]) for r in raw]);mono=np.empty_like(h);current=[]
                for i,(xp,pre) in enumerate(captured):
                    mx[i]=candidate_coordinates(draws[0][i,:,0],length)
                    mono[i]=interp0(np.arange(length),mx[i],pre).astype(np.float32)
                    d=np.diff(xp);md=np.diff(mx[i]);threshold=5*np.sqrt(np.mean(np.diff(data.series[i])**2))
                    r=dict(dataset=name,seed=seed,snapshot=snapshot,index=i,length=length,nonincreasing=int((d<=0).sum()),negative=int((d<0).sum()),duplicates=int((d==0).sum()),minimum_increment=float(d.min()),longest_nonincreasing=longest(d<=0),clipped=int(((before[i]<0)|(before[i]>length-1)).sum()),lower_boundary_fraction=float((xp==0).mean()),upper_boundary_fraction=float((xp==length-1).mean()),endpoint_order_valid=bool(xp[0]<xp[-1]),monotone_minimum_increment=float(md.min()),monotone_nonincreasing=int((md<=0).sum()),historical_displacement_mean=float(np.abs(xp-np.arange(length)).mean()/(length-1)),monotone_displacement_mean=float(np.abs(mx[i]-np.arange(length)).mean()/(length-1)),path_difference_mean=float(np.abs(xp-mx[i]).mean()/(length-1)),output_correlation=corr(h[i],mono[i]),output_mse=float(np.mean((h[i].astype(float)-mono[i])**2)),historical_prewarp_correlation=corr(pre,h[i]),monotone_prewarp_correlation=corr(pre,mono[i]))
                    for prefix,x in [('original',data.series[i]),('prewarp',pre),('historical',h[i]),('monotone',mono[i])]:r.update({prefix+'_'+k:v for k,v in signal(x,threshold).items()})
                    current.append(r)
                records.extend(current)
                np.savez_compressed(dest/(snapshot+'.npz'),historical_xp=hx,historical_unclipped=before,monotone_xp=mx,historical=h,monotone=mono,prewarp=np.stack([c[1] for c in captured]),multipliers=draws[0])
                s=dict(dataset=name,seed=seed,snapshot=snapshot,length=length,series=n,affected=sum(r['nonincreasing']>0 for r in current),negative_series=sum(r['negative']>0 for r in current),nonincreasing_intervals=sum(r['nonincreasing'] for r in current),intervals=n*(length-1),minimum_increment=min(r['minimum_increment'] for r in current),longest_nonincreasing=max(r['longest_nonincreasing'] for r in current),monotone_invalid=sum(r['monotone_nonincreasing']>0 for r in current),mean_output_mse=float(np.mean([r['output_mse'] for r in current])),mean_output_correlation=float(np.mean([r['output_correlation'] for r in current if r['output_correlation'] is not None])),speed_quantiles_historical=np.quantile(np.diff(hx,axis=1),[0,.01,.05,.5,.95,.99,1]).tolist(),speed_quantiles_monotone=np.quantile(np.diff(mx,axis=1),[0,.01,.05,.5,.95,.99,1]).tolist())
                summaries.append(s);print(name,seed,snapshot,s['affected']/n,flush=True)
    with (out/'series.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    for name,value in [('snapshots.json',summaries),('w0-references.json',references)]: (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print('DATA_ONLY_COMPLETE',len(summaries),len(records),flush=True)
if __name__=='__main__':main()
