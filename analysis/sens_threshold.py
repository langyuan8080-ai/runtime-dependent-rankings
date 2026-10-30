"""Sensitivity of the outcome classes to the practical-relevance threshold (exploratory, added after review).
Resolved = Holm p < 0.05 and |difference| >= max(t * faster median, 0.05 ms), for t = 1%, 2% (protocol) and 5%;
the Holm p values do not depend on t. Outcomes are re-derived relative to eager batch 1 as in rq1.py."""
import pandas as pd, numpy as np
P=pd.read_csv("rq1_pairs.csv"); out=[]
for t in (0.01,0.02,0.05):
    Q=P.copy(); Q["res"]=(Q.p_holm<0.05)&(Q["diff"].abs()>=np.maximum(t*np.minimum(Q.mx,Q.my),0.05))
    Q["sg"]=np.where(Q.res,np.sign(Q.mx-Q.my),0)
    ref=Q[(Q.batch==1)&(Q.runtime=="eager")].set_index(["x","y"])[["res","sg"]]
    O=Q[~((Q.batch==1)&(Q.runtime=="eager"))].copy()
    r=ref.loc[list(zip(O.x,O.y))].values; rr=r[:,0].astype(bool); rs=r[:,1]
    O["cat"]=np.select([rr&O.res&(rs!=O.sg),rr&~O.res,~rr&O.res],["strict reversal","lost advantage","emergent difference"],"unchanged")
    O["t"]=t; out.append(O[["t","batch","runtime","family","x","y","cat"]])
    g=O.groupby(["batch","runtime"]).cat.apply(lambda s:(s!="unchanged").sum())
    print(f"t={t:.0%}: resolved eager b1 {int(ref.res.sum())}, strict reversals {int((O.cat=='strict reversal').sum())} "
          f"(D2 {int(((O.cat=='strict reversal')&(O.family=='D2')).sum())}), changed pairs per setting {g.min()}-{g.max()}")
pd.concat(out).to_csv("sens_pairs_threshold.csv",index=False)
