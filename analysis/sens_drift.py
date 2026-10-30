"""Exploratory, added after external review (protocol deviation log, 30 Oct 2026).
Sensitivity of the eager batch-1 reference classification to between-block host drift.
Each eager batch-1 repetition is divided by the drift reference of its block interpolated in time (start/middle/end),
relative to the median reference over the core layer; pairs are then re-tested exactly as in rq1.py."""
import pandas as pd, numpy as np, itertools
rng=np.random.default_rng(20261016); B=20000
L=pd.read_csv("full_res/latency.csv"); L["t"]=pd.to_datetime(L.ts)
R=L[L.runtime=="ref_eager"]; g0=R.median_ms.median()
E=L[(L.runtime=="eager")&(L.batch==1)].copy()
def fac(row):
    r=R[R.block==row.cfg].sort_values("t")
    return np.interp(row.t.value, r.t.astype("int64").values, r.median_ms.values)/g0
E["f"]=E.apply(fac,axis=1); E["adj"]=E.median_ms/E.f
D3=["c1pos","c1bn","Bplain","Bg4","seg1","seg4"]
PAIRS=[("D1","p2_832","base960"),("D1","p2_960","base1120"),("D1b","c1pos","base960"),("D1b","safew288","base960")]
PAIRS+=[("D2",a,b) for a in ("c1pos","safew288") for b in ("y11m","y11l")]
PAIRS+=[("D3",a,b) for a,b in itertools.combinations(D3,2)]+[("D4","Awide","c1pos"),("D4","afterA","Bnarrow")]
def test(col):
    groups={c:[g[col].values for _,g in v.groupby("build")] for c,v in E.groupby("cfg")}
    cache={}
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
    T=pd.DataFrame(rows); o=np.argsort(T.p.values); m=len(T); adj=np.empty(m); run=0
    for i,k in enumerate(o): run=max(run,(m-i)*T.p.values[k]); adj[k]=min(1,run)
    T["p_holm"]=adj; T["resolved"]=(T.p_holm<0.05)&T.practical; T["sign"]=np.where(T.resolved,np.sign(T.mx-T.my),0)
    return T
A=test("median_ms"); N=test("adj")
print("block factors: min %.3f max %.3f"%(E.groupby('cfg').f.median().min(),E.groupby('cfg').f.median().max()))
C=A[["family","x","y","rel","resolved","sign"]].merge(N[["x","y","rel","resolved","sign"]],on=["x","y"],suffixes=("_orig","_norm"))
print("resolved eager b1: original",int(A.resolved.sum()),"normalized",int(N.resolved.sum()))
print(C[(C.resolved_orig!=C.resolved_norm)|(C.sign_orig!=C.sign_norm)].round(3).to_string())
# re-categorize all other settings against the normalized reference
P=pd.read_csv("rq1_pairs.csv")
ref=N.set_index(["x","y"])[["resolved","sign"]]
def cat(r):
    s=ref.loc[(r.x,r.y)]
    if s.resolved and r.resolved: return "unchanged" if s.sign==r.sign else "strict reversal"
    if s.resolved: return "lost advantage"
    if r.resolved: return "emergent difference"
    return "unchanged"
O=P[~((P.batch==1)&(P.runtime=="eager"))].copy(); O["cat_norm"]=O.apply(cat,axis=1)
print("strict reversals: original",int((O.category=="strict reversal").sum()),"normalized",int((O.cat_norm=="strict reversal").sum()))
print("settings with >=6 changed pairs (original / normalized):",
      int((O.groupby(['batch','runtime']).category.apply(lambda s:(s!='unchanged').sum())>=6).sum()),
      int((O.groupby(['batch','runtime']).cat_norm.apply(lambda s:(s!='unchanged').sum())>=6).sum()))
ch=O[O.category!=O.cat_norm]; print("changed pair-setting categories:",len(ch)); print(ch.groupby(['x','y','category','cat_norm']).size().to_string())
b1=O[O.batch==1].groupby('runtime').cat_norm.apply(lambda s:(s!='unchanged').sum()); print("changed pairs per batch-1 setting (normalized):",b1.min(),"-",b1.max())
E.groupby('cfg').f.median().round(3).to_csv('sens_block_factors.csv'); C.to_csv('sens_eager_b1_pairs_norm.csv',index=False)
O[['batch','runtime','family','x','y','category','cat_norm']].to_csv('sens_pairs_drift_norm.csv',index=False)
