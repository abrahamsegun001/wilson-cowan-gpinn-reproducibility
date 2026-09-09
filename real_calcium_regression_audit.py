#!/usr/bin/env python3
"""Real-calcium Wilson-Cowan pseudo-derivative regression audit."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SMOOTHING = [(1.0, 2.0), (2.0, 2.0), (2.0, 1.0)]
PRIMARY = (2.0, 2.0)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./data/processed')
    p.add_argument('--output_dir', default='./outputs/regression_audit')
    p.add_argument('--num_chunks', type=int, default=20)
    p.add_argument('--subject_prefix', default='sub-GC48_ses-day0')
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--l_fov', type=float, default=10.0)
    p.add_argument('--n_points', type=int, default=90000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eps', type=float, default=1e-4)
    p.add_argument('--gpinn_results', default='./outputs/temporal_tracking_results.npz')
    return p.parse_args()

def fpath(data_dir,prefix,i):
    return data_dir/f'{prefix}_chunk-{i:02d}_dfof.npz'

def global_minmax(data_dir,prefix,n):
    gmin,gmax=np.inf,-np.inf; found=0
    for i in range(n):
        f=fpath(data_dir,prefix,i)
        if not f.exists(): continue
        E=np.load(f)['dfof']
        gmin=min(gmin,float(np.nanmin(E))); gmax=max(gmax,float(np.nanmax(E))); found+=1
    if found==0: raise FileNotFoundError('No chunk files found')
    return gmin,gmax,found

def sample_idx(shape,n,seed):
    T,H,W=shape; total=(T-2)*(H-2)*(W-2); n=min(n,total)
    rng=np.random.default_rng(seed); flat=rng.choice(total,size=n,replace=False)
    hw=(H-2)*(W-2); ti=flat//hw+1; rem=flat%hw; yi=rem//(W-2)+1; xi=rem%(W-2)+1
    return ti,yi,xi

def derivatives(E,ti,yi,xi,fps,l_fov):
    _,H,W=E.shape; dt=1.0/fps; dx=l_fov/(W-1); dy=l_fov/(H-1)
    E0=E[ti,yi,xi].astype(np.float64)
    Et=(E[ti+1,yi,xi]-E[ti-1,yi,xi])/(2*dt)
    d2x=(E[ti,yi,xi+1]-2*E0+E[ti,yi,xi-1])/(dx*dx)
    d2y=(E[ti,yi+1,xi]-2*E0+E[ti,yi-1,xi])/(dy*dy)
    return E0,Et.astype(np.float64),(d2x+d2y).astype(np.float64)

def fit_reg(E,Et,lap,tau,eps):
    q=tau*Et+E
    finite=np.isfinite(E)&np.isfinite(Et)&np.isfinite(lap)&np.isfinite(q)
    valid=finite&(q>eps)&(q<1-eps)
    out={
        'n_total':int(len(E)),'n_finite':int(finite.sum()),'n_valid':int(valid.sum()),
        'fraction_valid':float(valid.mean()),
        'q_p01':float(np.quantile(q[finite],.01)) if finite.any() else np.nan,
        'q_p50':float(np.quantile(q[finite],.50)) if finite.any() else np.nan,
        'q_p99':float(np.quantile(q[finite],.99)) if finite.any() else np.nan,
    }
    if valid.sum()<100:
        out.update(success=False,failure_reason='fewer than 100 valid points'); return out
    Ev,Lv,qv,Etv=E[valid],lap[valid],q[valid],Et[valid]
    y=np.log(qv/(1-qv)); X=np.column_stack([Ev,Lv,np.ones_like(Ev)])
    fit=lsq_linear(X,y,bounds=(np.array([0.5,0.1,-np.inf]),np.array([6.0,5.0,np.inf])),method='trf',lsmr_tol='auto')
    w,D,h=map(float,fit.x); yhat=X@fit.x
    Xstd=np.column_stack([(Ev-Ev.mean())/(Ev.std()+1e-12),(Lv-Lv.mean())/(Lv.std()+1e-12)])
    z=w*Ev+D*Lv+h; rhs=-Ev+1/(1+np.exp(-np.clip(z,-40,40))); pres=tau*Etv-rhs
    out.update(
        success=bool(fit.success),D_E=D,w_EE=w,h=h,cost=float(fit.cost),optimality=float(fit.optimality),
        logit_rmse=float(np.sqrt(np.mean((yhat-y)**2))),
        pde_residual_rmse=float(np.sqrt(np.mean(pres**2))),
        condition_number_raw_X=float(np.linalg.cond(X)),
        condition_number_standardized_E_lap=float(np.linalg.cond(Xstd)),
        corr_E_laplacian=float(np.corrcoef(Ev,Lv)[0,1]),
        E_mean=float(Ev.mean()),E_sd=float(Ev.std()),Et_sd=float(Etv.std()),lap_sd=float(Lv.std()),
        D_at_lower_bound=bool(D<=0.1005),D_at_upper_bound=bool(D>=4.9995),
        w_at_lower_bound=bool(w<=0.5005),w_at_upper_bound=bool(w>=5.9995))
    return out

def write_csv(path,rows):
    if not rows: return
    fields=sorted({k for r in rows for k in r})
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def load_gpinn(path):
    if not path.exists(): return None
    d=np.load(path,allow_pickle=True)
    if not all(k in d for k in ('D_E','w_EE','h')): return None
    return {k:np.asarray(d[k],float) for k in ('D_E','w_EE','h')}

def plot_compare(rows,gpinn,param,out):
    c=np.array([r['chunk_idx'] for r in rows]); reg=np.array([r[param] for r in rows])
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.plot(c,reg,marker='o',label='Pseudo-derivative regression')
    if gpinn is not None:
        n=min(len(c),len(gpinn[param])); ax.plot(c[:n],gpinn[param][:n],marker='s',label='Existing gPINN')
    ax.set_xlabel('30-s chunk'); ax.set_ylabel(param); ax.set_title(f'{param}: regression audit vs existing gPINN'); ax.legend(); fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)

def main():
    a=parse_args(); data_dir=Path(a.data_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    gmin,gmax,found=global_minmax(data_dir,a.subject_prefix,a.num_chunks); scale=gmax-gmin+1e-8
    print(f'Found {found} chunks; global min={gmin:.8g}, max={gmax:.8g}',flush=True)
    rows=[]
    for i in range(a.num_chunks):
        f=fpath(data_dir,a.subject_prefix,i)
        if not f.exists(): print(f'Chunk {i:02d} missing',flush=True); continue
        raw=np.load(f)['dfof'].astype(np.float64); E=(raw-gmin)/scale; ti,yi,xi=sample_idx(E.shape,a.n_points,a.seed+i)
        print(f'\nChunk {i:02d} shape={E.shape}',flush=True)
        for st,sxy in SMOOTHING:
            Es=gaussian_filter(E,sigma=(st,sxy,sxy),mode='reflect')
            E0,Et,lap=derivatives(Es,ti,yi,xi,a.fps,a.l_fov); reg=fit_reg(E0,Et,lap,a.tau,a.eps)
            row={'chunk_idx':i,'time_start_min':0.5*i,'smooth_sigma_t_frames':st,'smooth_sigma_space_pixels':sxy,'source_file':f.name,**reg}; rows.append(row)
            if reg.get('success'):
                print(f"  smooth=({st:g},{sxy:g}) valid={reg['fraction_valid']:.3f} cond={reg['condition_number_standardized_E_lap']:.2f} D={reg['D_E']:.4f} w={reg['w_EE']:.4f} h={reg['h']:.4f} PDE_RMSE={reg['pde_residual_rmse']:.3e}",flush=True)
            else: print(f"  smooth=({st:g},{sxy:g}) FAILED",flush=True)
    write_csv(out/'real_regression_all_settings.csv',rows)
    primary=[r for r in rows if r.get('success') and r['smooth_sigma_t_frames']==PRIMARY[0] and r['smooth_sigma_space_pixels']==PRIMARY[1]]
    write_csv(out/'real_regression_primary.csv',primary)
    stab=[]
    for c in sorted({r['chunk_idx'] for r in rows}):
        sub=[r for r in rows if r['chunk_idx']==c and r.get('success')]
        if not sub: continue
        def A(k): return np.array([r[k] for r in sub],float)
        D,w,h=A('D_E'),A('w_EE'),A('h')
        stab.append({'chunk_idx':c,'n_successful_smoothing_settings':len(sub),
                     'D_E_mean':float(D.mean()),'D_E_sd':float(D.std(ddof=1)) if len(D)>1 else 0.0,'D_E_range':float(D.max()-D.min()),
                     'w_EE_mean':float(w.mean()),'w_EE_sd':float(w.std(ddof=1)) if len(w)>1 else 0.0,'w_EE_range':float(w.max()-w.min()),
                     'h_mean':float(h.mean()),'h_sd':float(h.std(ddof=1)) if len(h)>1 else 0.0,'h_range':float(h.max()-h.min()),
                     'valid_fraction_min':float(min(r['fraction_valid'] for r in sub)),
                     'condition_number_standardized_max':float(max(r['condition_number_standardized_E_lap'] for r in sub))})
    write_csv(out/'real_regression_smoothing_stability.csv',stab)
    gp=load_gpinn(Path(a.gpinn_results))
    if primary:
        for p in ('D_E','w_EE','h'): plot_compare(primary,gp,p,out/f'compare_{p}.png')
    summary={'purpose':'real-calcium pseudo-derivative regression audit','global_min':gmin,'global_max':gmax,'num_chunks_found':found,
             'normalization':'Session-global min-max scaling, mirroring the current gPINN for like-for-like comparison.',
             'primary_smoothing':{'sigma_t_frames':2.0,'sigma_space_pixels':2.0},
             'sensitivity_settings':[{'sigma_t_frames':x,'sigma_space_pixels':y} for x,y in SMOOTHING],
             'fps':a.fps,'tau':a.tau,'l_fov_mm':a.l_fov,'n_points':a.n_points,
             'bounds':{'D_E':[0.1,5.0],'w_EE':[0.5,6.0],'h':'unbounded'},
             'gpinn_comparison_file_found':bool(gp is not None),
             'caution':'Real calcium fluorescence is a proxy for Wilson-Cowan activity. Interpret regression with valid fraction, conditioning, smoothing stability, and residual diagnostics.'}
    (out/'real_regression_audit_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'\nAudit complete: {out}',flush=True)
if __name__=='__main__': main()
