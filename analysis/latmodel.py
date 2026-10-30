"""Descriptive overlap model: T_wall ~ max(N_kernel * t_d, T_dev). t_d fitted on core eager cells (RTX 4090,
MPS off), then applied without refitting to Slice A eager cells (MPS 100/50/25%, inputs 640/960/1280)."""
import pandas as pd, numpy as np
C=pd.read_csv('rq2_cells.csv'); C=C[C.status=='ok']
def fit(x):
    tds=np.arange(5,25,0.05); err=[np.median(np.abs(np.maximum(x.n_kernel*t/1000,x.t_dev_ms)/x.t_wall-1)) for t in tds]
    return tds[int(np.argmin(err))]
def report(name,x,td):
    p=np.maximum(x.n_kernel*td/1000,x.t_dev_ms); e=np.abs(p/x.t_wall-1)*100
    pa=x.n_kernel*td/1000+x.t_dev_ms; ea=np.abs(pa/x.t_wall-1)*100
    pred_host=(x.n_kernel*td/1000>=2*x.t_dev_ms); obs_host=(x.h>=0.5)
    print(f"{name:34s} n={len(x):3d} t_d={td:5.2f}us  max-model err median {np.median(e):4.1f}% p90 {np.percentile(e,90):4.1f}% max {e.max():5.1f}% | additive median {np.median(ea):4.1f}% | regime agreement {(pred_host==obs_host).mean()*100:.0f}%")
    return td
out={}
for rt in ['eager','compile_default','ort_iobinding']:
    x=C[C.runtime==rt].dropna(subset=['t_wall']); td=fit(x); out[rt]=report(f"core {rt}",x,td)
    # leave-one-configuration-out
    errs=[]
    for c in x.cfg.unique():
        tr,te=x[x.cfg!=c],x[x.cfg==c]; t=fit(tr); errs+=list(np.abs(np.maximum(te.n_kernel*t/1000,te.t_dev_ms)/te.t_wall-1)*100)
    print(f"   leave-one-config-out error median {np.median(errs):.1f}% max {np.max(errs):.1f}%")
# Slice A: profile stage A + latency_A eager (median over reps/passes), applied with core eager t_d
P=pd.read_csv('extra_res/profile.csv'); P=P[(P.stage=='A')&(P.runtime=='eager')&(P.status=='ok')]
A=pd.read_csv('extra_res/latency_A.csv'); A=A[(A.runtime=='eager')&(A.status=='ok')&(A['pass']!='check')]
W=A.groupby(['cfg','imgsz','batch','mps']).median_ms.median().rename('t_wall').reset_index()
P['mps']=P.mps.astype(str); W['mps']=W.mps.astype(str)
M=P.merge(W,on=['cfg','imgsz','batch','mps']); M['h']=1-M.t_dev_ms/M.t_wall
print("Slice A eager cells matched:",len(M), sorted(M.mps.unique()))
report("Slice A eager, core t_d (no refit)",M,out['eager'])
for m in sorted(M.mps.unique()): report(f"   MPS {m}",M[M.mps==m],out['eager'])
# Without a profiler: T_dev replaced by the eager + CUDA Graph latency of the same configuration and batch (h < 0.04
# in that cell), t_d kept at the core eager value; regimes still judged against the profiled host share.
G=C[C.runtime=='eager_graph'][['cfg','batch','t_wall']].rename(columns={'t_wall':'t_graph'})
X=C[C.runtime=='eager'].dropna(subset=['t_wall']).merge(G,on=['cfg','batch'])
print("eager + CUDA Graph host share: max |h| %.3f"%C[C.runtime=='eager_graph'].h.abs().max())
report("core eager, T_dev := graph latency",X.assign(t_dev_ms=X.t_graph),out['eager'])
# leave-one-configuration-out: t_d fitted on the other configurations (graph latency as T_dev), applied to the held-out one
Xg=X.assign(t_dev_ms=X.t_graph); errs=[]; agree=[]
for c in Xg.cfg.unique():
    tr,te=Xg[Xg.cfg!=c],Xg[Xg.cfg==c]; t=fit(tr); p=np.maximum(te.n_kernel*t/1000,te.t_dev_ms)
    errs+=list(np.abs(p/te.t_wall-1)*100); agree+=list((te.n_kernel*t/1000>=2*te.t_dev_ms)==(te.h>=0.5))
print(f"   leave-one-config-out (graph latency as T_dev): error median {np.median(errs):.1f}% max {np.max(errs):.1f}%, regime agreement {np.mean(agree)*100:.0f}%")
