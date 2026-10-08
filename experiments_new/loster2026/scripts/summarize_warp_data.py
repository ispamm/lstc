import sys,json,ast
from pathlib import Path
import numpy as np
import pandas as pd
local=Path(r'G:\Articoli\Articoli da Completare\LoSTer 2026\warp-validation')
repo=Path(r'H:\Il mio Drive\Articoli\LoSTer')
a=ast.parse((repo/'models/LoSTer/data/augmentation.py').read_text())
b=ast.parse((local/'sources/iwana-uchida-augmentation.py').read_text())
fn=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='time_warp'),include_attributes=False)
print('EXACT_TIME_WARP_AST_EQUAL',fn(a)==fn(b))
if not (local/'data-only/series.csv').exists():print('DATA_ONLY_PENDING');sys.exit()
d=pd.read_csv(local/'data-only/series.csv');s=json.loads((local/'data-only/snapshots.json').read_text())
print('TOTAL',len(d),'INVALID',int((d.nonincreasing>0).sum()),'NEGATIVE',int((d.negative>0).sum()),'INTERVALS',int(d.nonincreasing.sum()),'FRAC',d.nonincreasing.sum()/(d.length-1).sum(),'MIN',d.minimum_increment.min(),'LONGEST',d.longest_nonincreasing.max())
for name,g in d.groupby('dataset',sort=False):print(name,len(g),round((g.nonincreasing>0).mean(),6),round((g.negative>0).mean(),6),round(g.nonincreasing.sum()/(g.length-1).sum(),6),round(g.output_correlation.mean(),6),round(g.output_mse.mean(),6))
cols=[c for c in d.columns if d[c].dtype.kind in 'fi' and c not in ('seed','index')]
summary=d.groupby('dataset',sort=False)[cols].mean();summary['series']=d.groupby('dataset',sort=False).size();summary['invalid_fraction']=d.groupby('dataset',sort=False).nonincreasing.apply(lambda x:(x>0).mean());summary['negative_fraction']=d.groupby('dataset',sort=False).negative.apply(lambda x:(x>0).mean())
summary.to_csv(repo/'analysis/experiments/warp-dataset-summary.csv')
for dimension in ['dataset','seed','snapshot','length']:
    d.groupby(dimension,sort=False)[cols if dimension not in cols else [c for c in cols if c!=dimension]].agg(['mean','min','max']).to_csv(local/('by-'+dimension+'.csv'))
result={'series':len(d),'invalid_series':int((d.nonincreasing>0).sum()),'invalid_fraction':float((d.nonincreasing>0).mean()),'negative_series':int((d.negative>0).sum()),'negative_fraction':float((d.negative>0).mean()),'intervals':int((d.length-1).sum()),'nonincreasing_intervals':int(d.nonincreasing.sum()),'minimum_increment':float(d.minimum_increment.min()),'longest_nonincreasing':int(d.longest_nonincreasing.max()),'mean_output_correlation':float(d.output_correlation.mean()),'mean_output_mse':float(d.output_mse.mean()),'monotone_invalid':int((d.monotone_nonincreasing>0).sum()),'mean_path_difference':float(d.path_difference_mean.mean()),'max_monotone_minimum_increment':float(d.monotone_minimum_increment.max()),'min_monotone_increment':float(d.monotone_minimum_increment.min())}
(local/'data-only-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
