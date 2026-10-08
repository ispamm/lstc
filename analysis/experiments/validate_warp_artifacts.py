"""Independent array-based incidence verification; no model training or labels."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
repo=Path(__file__).resolve().parents[2]
out=Path(r'G:\Articoli\Articoli da Completare\LoSTer 2026\warp-validation')
def hash_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    d=pd.read_csv(out/'data-only/series.csv');snapshots=json.loads((out/'data-only/snapshots.json').read_text());independent=[]
    for saved in snapshots:
        name,seed,snap=saved['dataset'],saved['seed'],saved['snapshot']
        with np.load(out/'data-only'/name/('seed-%d'%seed)/(snap+'.npz')) as arrays:
            hx=arrays['historical_xp'];mx=arrays['monotone_xp'];h=arrays['historical'];m=arrays['monotone'];raw=arrays['historical_unclipped']
            n,length=hx.shape;diff=np.diff(hx,axis=1);mdiff=np.diff(mx,axis=1)
            assert hx.shape==mx.shape==h.shape==m.shape and np.isfinite(mx).all() and np.isfinite(h).all() and np.isfinite(m).all()
            assert np.all(mx[:,0]==0) and np.all(mx[:,-1]==length-1) and np.all(mdiff>0)
            mp=out.parent/'pilot/artifacts'/name/'LoSTer-Legacy-Clean'/('seed-%d'%seed)/'manifest.json'
            old=json.loads(mp.read_text());assert hash_array(h)==old['augmentation'][snap]['snapshot_sha256']
            g=d[(d.dataset==name)&(d.seed==seed)&(d.snapshot==snap)].sort_values('index')
            np.testing.assert_array_equal((diff<=0).sum(1),g.nonincreasing)
            np.testing.assert_array_equal((diff<0).sum(1),g.negative)
            np.testing.assert_array_equal((diff==0).sum(1),g.duplicates)
            np.testing.assert_allclose(diff.min(1),g.minimum_increment,rtol=0,atol=1e-12)
            np.testing.assert_allclose(((raw<0)|(raw>length-1)).sum(1),g.clipped,rtol=0,atol=0)
            np.testing.assert_allclose(np.mean((h.astype(float)-m.astype(float))**2,axis=1),g.output_mse,rtol=1e-12,atol=1e-12)
            r=dict(dataset=name,seed=seed,snapshot=snap,length=length,total_series=n,invalid_series=int(np.any(diff<=0,axis=1).sum()),invalid_fraction=float(np.any(diff<=0,axis=1).mean()),reversed_series=int(np.any(diff<0,axis=1).sum()),reversed_fraction=float(np.any(diff<0,axis=1).mean()),nonincreasing_intervals=int((diff<=0).sum()),interval_fraction=float((diff<=0).mean()),minimum_increment=float(diff.min()),most_negative_increment=float(diff[diff<0].min()) if np.any(diff<0) else None,longest_nonincreasing_region=int(g.longest_nonincreasing.max()),duplicate_intervals=int((diff==0).sum()),duplicate_interval_fraction=float((diff==0).mean()),duplicated_series_fraction=float(np.any(diff==0,axis=1).mean()),clipped_coordinates=int(((raw<0)|(raw>length-1)).sum()),clipped_coordinate_fraction=float(((raw<0)|(raw>length-1)).mean()),clipped_series_fraction=float(np.any((raw<0)|(raw>length-1),axis=1).mean()),lower_boundary_fraction=float((hx==0).mean()),upper_boundary_fraction=float((hx==length-1).mean()),all_endpoints_ordered=bool(np.all(hx[:,0]<hx[:,-1])),monotone_minimum_increment=float(mdiff.min()),monotone_invalid_series=0)
            levels=[0,.01,.05,.5,.95,.99,1]
            r['historical_forward_speed_quantiles']=json.dumps(np.quantile(diff,levels).tolist())
            r['monotone_forward_speed_quantiles']=json.dumps(np.quantile(mdiff,levels).tolist())
            historical_sampling=np.stack([np.diff(np.interp(np.arange(length),xp,np.arange(length))) for xp in hx])
            monotone_sampling=np.stack([np.diff(np.interp(np.arange(length),xp,np.arange(length))) for xp in mx])
            r['historical_operational_sampling_step_quantiles']=json.dumps(np.quantile(historical_sampling,levels).tolist())
            r['monotone_inverse_sampling_speed_quantiles']=json.dumps(np.quantile(monotone_sampling,levels).tolist())
            independent.append(r)
        print('ARRAYS_VERIFIED',name,seed,snap,flush=True)
    tables=repo/'analysis/experiments/warp';tables.mkdir(exist_ok=True)
    pd.DataFrame(independent).to_csv(tables/'snapshot-severity.csv',index=False)
    for dim in ['dataset','seed','snapshot','length']:
        rows=[]
        for key,g in d.groupby(dim,sort=False):
            rows.append({dim:key,'generated_series':len(g),'invalid_series':int((g.nonincreasing>0).sum()),'invalid_fraction':float((g.nonincreasing>0).mean()),'reversed_series':int((g.negative>0).sum()),'reversed_fraction':float((g.negative>0).mean()),'intervals':int((g.length-1).sum()),'nonincreasing_intervals':int(g.nonincreasing.sum()),'interval_fraction':float(g.nonincreasing.sum()/(g.length-1).sum()),'minimum_increment':float(g.minimum_increment.min()),'longest_nonincreasing_region':int(g.longest_nonincreasing.max()),'duplicate_interval_fraction':float(g.duplicates.sum()/(g.length-1).sum()),'clipped_coordinate_fraction':float(g.clipped.sum()/g.length.sum()),'lower_boundary_fraction':float(np.average(g.lower_boundary_fraction,weights=g.length)),'upper_boundary_fraction':float(np.average(g.upper_boundary_fraction,weights=g.length)),'all_endpoints_ordered':bool(g.endpoint_order_valid.all())})
        pd.DataFrame(rows).to_csv(tables/('severity-by-'+dim+'.csv'),index=False)
    examples=[]
    for name,g in d.groupby('dataset',sort=False):
        fixed=g[(g.seed==0)&(g.snapshot=='A_train')].sort_values('index')
        for index in [0,len(fixed)//2,len(fixed)-1]:
            r=fixed[fixed['index']==index].iloc[0].to_dict();r['selection']='fixed-index representative';examples.append(r)
        r=g.sort_values(['minimum_increment','seed','snapshot','index']).iloc[0].to_dict();r['selection']='worst-case across all ten dataset snapshots';examples.append(r)
    pd.DataFrame(examples).to_csv(tables/'diagnostic-examples.csv',index=False)
    result=dict(snapshots=80,series=int(sum(r['total_series'] for r in independent)),invalid_series=int(sum(r['invalid_series'] for r in independent)),reversed_series=int(sum(r['reversed_series'] for r in independent)),all_monotone_paths_strict=True,all_historical_hashes_exact=True,all_series_counts_mse_and_clipping_crosschecked=True)
    (out/'independent-data-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
