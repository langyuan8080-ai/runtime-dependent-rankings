"""Exploratory, added after external review (protocol deviation log, 30 Oct 2026).
Sensitivity of the outcome classes to between-block host drift. Every configuration was measured in its own block, and the
drift reference (baseline, eager, batch 1) was timed at the start, middle and end of each block. Each batch-1 repetition is
divided by a drift factor: the block's reference interpolated in time, relative to the median reference over the core layer,
scaled by the cell's host share relative to the eager host share of that configuration (h_cell / h_eager; 1 for eager,
about 0 for CUDA Graph cells). Batch 8 and 32 cells are device-bound (h <= 0.05) and are left as measured.
Pairs are re-tested exactly as in rq1.py and re-classified against the normalized eager batch-1 reference.
(a) reference only: only eager batch 1 normalized; (b) all batch-1 cells normalized."""
import pandas as pd, numpy as np, itertools
rng=np.random.default_rng(20261016); B=20000
L=pd.read_csv("full_res/latency.csv"); L["t"]=pd.to_datetime(L.ts)
R=L[L.runtime=="ref_eager"]; g0=R.median_ms.median()
H=pd.read_csv("rq2_cells.csv"); H=H[H.batch==1].set_index(["cfg","runtime"]).h.clip(lower=0)
X=L[(L.batch==1)&(L.runtime!="ref_eager")&(L.runtime!="eager_unfused")].copy()
ref={c:(r.t.astype("int64").values, r.median_ms.values) for c,r in R.sort_values("t").groupby("block")}
X["f"]=[np.interp(t.value,*ref[c])/g0 for c,t in zip(X.cfg,X.t)]
X["w"]=[H.get((c,r),np.nan)/H[(c,"eager")] for c,r in zip(X.cfg,X.runtime)]
X["adj"]=X.median_ms/(1+X.w*(X.f-1))
D3=["c1pos","c1bn","Bplain","Bg4","seg1","seg4"]
PAIRS=[("D1","p2_832","base960"),("D1","p2_960","base1120"),("D1b","c1pos","base960"),("D1b","safew288","base960")]
PAIRS+=[("D2",a,b) for a in ("c1pos","safew288") for b in ("y11m","y11l")]
PAIRS+=[("D3",a,b) for a,b in itertools.combinations(D3,2)]+[("D4","Awide","c1pos"),("D4","afterA","Bnarrow")]
def holm(p):
    o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,k in enumerate(o): run=max(run,(m-i)*p[k]); adj[k]=min(1,run)
    return adj
def test(E,col):
    groups={c:[g[col].values for _,g in v.groupby("build")] for c,v in E.groupby("cfg")}; cache={}
    def bs(c):
        if c not in cache:
            gs=groups[c]; nb=len(gs); out=np.empty(B); bi=rng.integers(0,nb,(B,nb))
            for j in range(B): out[j]=np.median(np.concatenate([gs[b][rng.integers(0,len(gs[b]),len(gs[b]))] for b in bi[j]]))
            cache[c]=out
        return cache[c]
    rows=[]
    for fam,x,y in PAIRS:
        mx,my=np.median(np.concatenate(groups[x])),np.median(np.concatenate(groups[y])); d=bs(x)-bs(y)
        p=max(min(1,2*min((d<=0).mean(),(d>=0).mean())),1/B)
        rows.append(dict(family=fam,x=x,y=y,mx=mx,my=my,rel=(mx-my)/min(mx,my),p=p,practical=abs(mx-my)>=max(0.02*min(mx,my),0.05)))
    T=pd.DataFrame(rows); T["p_holm"]=holm(T.p.values); T["resolved"]=(T.p_holm<0.05)&T.practical
    T["sign"]=np.where(T.resolved,np.sign(T.mx-T.my),0); return T
E=X[X.runtime=="eager"]
A=test(E,"median_ms"); N=test(E,"adj")
print("drift factors per configuration (median): %.3f-%.3f; block reference medians %.3f-%.3f ms"%(
      E.groupby("cfg").f.median().min(),E.groupby("cfg").f.median().max(),R.groupby("block").median_ms.median().min(),R.groupby("block").median_ms.median().max()))
C=A[["family","x","y","rel","resolved","sign"]].merge(N[["x","y","rel","resolved","sign"]],on=["x","y"],suffixes=("_orig","_norm"))
print("resolved eager b1: original",int(A.resolved.sum()),"normalized",int(N.resolved.sum()))
print(C[(C.resolved_orig!=C.resolved_norm)|(C.sign_orig!=C.sign_norm)].round(3).to_string())
refN=N.set_index(["x","y"])[["resolved","sign"]]
def cat(s,res,sg):
    if s.resolved and res: return "unchanged" if s.sign==sg else "strict reversal"
    if s.resolved: return "lost advantage"
    if res: return "emergent difference"
    return "unchanged"
P=pd.read_csv("rq1_pairs.csv"); O=P[~((P.batch==1)&(P.runtime=="eager"))].copy()
O["cat_norm_ref"]=[cat(refN.loc[(r.x,r.y)],r.resolved,r.sign) for r in O.itertuples()]
O["cat_norm_all"]=O.cat_norm_ref
for rt in sorted(X.runtime.unique()):
    if rt=="eager": continue
    T=test(X[X.runtime==rt],"adj").set_index(["x","y"]); m=(O.batch==1)&(O.runtime==rt)
    O.loc[m,"cat_norm_all"]=[cat(refN.loc[(r.x,r.y)],T.loc[(r.x,r.y)].resolved,T.loc[(r.x,r.y)].sign) for r in O[m].itertuples()]
for c in ["category","cat_norm_ref","cat_norm_all"]:
    g=O.groupby(["batch","runtime"])[c].apply(lambda s:(s!="unchanged").sum())
    print(f"{c:13s} strict reversals {int((O[c]=='strict reversal').sum()):3d}; changed pairs per setting {g.min()}-{g.max()}; "
          f"batch-1 settings {g.loc[1].min()}-{g.loc[1].max()}")
ch=O[O.category!=O.cat_norm_ref]; print("strict-reversal changes (reference only):")
print(ch[(ch.category=="strict reversal")|(ch.cat_norm_ref=="strict reversal")].groupby(["x","y","batch","category","cat_norm_ref"]).size().to_string())
d=O[O.cat_norm_all!=O.cat_norm_ref]; print("additional changes when all batch-1 cells are normalized:",len(d))
if len(d): print(d.groupby(["runtime","x","y","cat_norm_ref","cat_norm_all"]).size().to_string())
E.groupby("cfg").f.median().round(3).to_csv("sens_block_factors.csv"); C.to_csv("sens_eager_b1_pairs_norm.csv",index=False)
O[["batch","runtime","family","x","y","category","cat_norm_ref","cat_norm_all"]].to_csv("sens_pairs_drift_norm.csv",index=False)
