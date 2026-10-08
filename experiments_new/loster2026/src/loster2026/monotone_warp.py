"""Report 07 M1: positive log-speed integration; methodological validity correction.
HistoricalWarp remains in augmentation.augment unchanged. No labels or rescue draws.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from .augmentation import array_hash

def monotone_coordinates(multipliers, length):
    m=np.asarray(multipliers,dtype=np.float64)
    if length<2 or m.ndim!=1 or len(m)<2 or not np.isfinite(m).all():
        raise ValueError('Finite knots and L>=2 required')
    steps=np.arange(length)
    z=CubicSpline(np.linspace(0,length-1,len(m)),m-1)(steps)
    speed=np.exp(z-z.max())
    c=np.r_[0.,np.cumsum((speed[:-1]+speed[1:])/2)]
    if not np.isfinite(c[-1]) or c[-1]<=0:raise ValueError('Invalid integrated speed')
    xp=c*((length-1)/c[-1]);xp[-1]=length-1
    if not np.isfinite(xp).all() or not np.all(np.diff(xp)>0):
        raise ValueError('Unrepresentable monotone coordinates; no silent repair')
    return xp

def augment_monotone(series,config):
    x=np.asarray(series,dtype=np.float64)[:,:,None]
    if not np.isfinite(x).all() or x.shape[1]<2:raise ValueError('Invalid series')
    rng=np.random
    flip=rng.choice([-1,1],size=(x.shape[0],x.shape[2]))
    axis=np.arange(x.shape[2]);rng.shuffle(axis)
    x=flip[:,None,:]*x[:,:,axis]
    steps=np.arange(x.shape[1])
    segments=rng.randint(1,config.max_segments_exclusive,size=x.shape[0])
    permuted=np.zeros_like(x)
    for i,row in enumerate(x):
        if segments[i]>1:
            splits=np.array_split(steps,segments[i]);order=rng.permutation(len(splits))
            permuted[i]=row[np.concatenate([splits[j] for j in order])]
        else:permuted[i]=row
    multipliers=rng.normal(1.,config.warp_sigma,size=(len(x),config.warp_knots+2,1))
    ret=np.empty_like(x);minima=[]
    for i,row in enumerate(permuted):
        xp=monotone_coordinates(multipliers[i,:,0],x.shape[1])
        ret[i,:,0]=np.interp(steps,xp,row[:,0]);minima.append(float(np.diff(xp).min()))
    a=ret[:,:,0].astype(np.float32)
    if not np.isfinite(a).all():raise ValueError('Nonfinite output; no silent repair')
    return a,dict(mode='MonotoneWarp',definition='positive-cubic-log-speed-trapezoid-v1',snapshot_sha256=array_hash(a),finite=True,
        minimum_coordinate_difference=min(minima),row_minimum_coordinate_differences=minima,
        nonincreasing_steps_per_row=[0]*len(x),nonincreasing_step_fraction=0.,affected_series_fraction=0.,
        clipped_coordinates_per_row=[0]*len(x),endpoint_valid=True)
