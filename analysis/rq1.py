import pandas as pd, numpy as np, itertools
from scipy import stats
pd.set_option("display.width",250); pd.set_option("display.max_rows",400); pd.set_option("display.max_columns",40)
rng=np.random.default_rng(20261016); B=20000
L=pd.read_csv("full_res/latency.csv"); L=L[L.runtime!="ref_eager"].copy()
GF={"base640":21.7,"p2_640":29.7,"afterA":31.0,"c1bn":36.9,"c1pos":50.3,"Awide":52.1,"Bnarrow":31.1,"Bplain":59.9,"Bg4":37.3,
    "seg1":59.9,"seg4":37.3,"safew288":63.4,"y11m":68.5,"y11l":87.6,"base960":48.9,"base1024":55.6,"base1120":66.5,"p2_832":50.1,"p2_960":66.7}
D3=["c1pos","c1bn","Bplain","Bg4","seg1","seg4"]
PAIRS=[("D1","p2_832","base960"),("D1","p2_960","base1120"),("D1b","c1pos","base960"),("D1b","safew288","base960")]
PAIRS+=[("D2",a,b) for a in ("c1pos","safew288") for b in ("y11m","y11l")]
PAIRS+=[("D3",a,b) for a,b in itertools.combinations(D3,2)]
PAIRS+=[("D4","Awide","c1pos"),("D4","afterA","Bnarrow")]
groups={k:[g.median_ms.values for _,g in v.groupby("build")] for k,v in L.groupby(["cfg","batch","runtime"])}
def boot(gs):
    # two-stage bootstrap of the median: builds with replacement, then repetitions within each build
    nb=len(gs); out=np.empty(B)
    bi=rng.integers(0,nb,(B,nb))
    for j in range(B):
        vals=np.concatenate([gs[b][rng.integers(0,len(gs[b]),len(gs[b]))] for b in bi[j]])
        out[j]=np.median(vals)
    return out
cache={}
def bs(k):
    if k not in cache: cache[k]=boot(groups[k])
    return cache[k]
rows=[]
settings=sorted({(b,r) for _,b,r in groups})
for b,r in settings:
    for fam,x,y in PAIRS:
        kx,ky=(x,b,r),(y,b,r)
        mx,my=np.median(np.concatenate(groups[kx])),np.median(np.concatenate(groups[ky]))
        d=bs(kx)-bs(ky); diff=mx-my
        p=min(1.0,2*min((d<=0).mean(),(d>=0).mean())); p=max(p,1/B)
        lo,hi=np.percentile(d,[2.5,97.5])
        rows.append(dict(batch=b,runtime=r,family=fam,x=x,y=y,mx=mx,my=my,diff=diff,rel=diff/min(mx,my),lo=lo,hi=hi,p=p,
                         practical=abs(diff)>=max(0.02*min(mx,my),0.05)))
R=pd.DataFrame(rows)
def holm(p):
    o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,k in enumerate(o): run=max(run,(m-i)*p[k]); adj[k]=min(1,run)
    return adj
R["p_holm"]=R.groupby(["batch","runtime"]).p.transform(lambda s: holm(s.values))
R["resolved"]=(R.p_holm<0.05)&R.practical
R["sign"]=np.where(R.resolved,np.sign(R["diff"]).astype(int),0)
S1=R[(R.batch==1)&(R.runtime=="eager")].set_index(["x","y"])[["resolved","sign"]]
def cat(row):
    s1=S1.loc[(row.x,row.y)]
    if s1.resolved and row.resolved: return "unchanged" if s1.sign==row.sign else "strict reversal"
    if s1.resolved and not row.resolved: return "lost advantage"
    if not s1.resolved and row.resolved: return "emergent difference"
    return "unchanged"
R["category"]=R.apply(cat,axis=1)
R.to_csv("rq1_pairs.csv",index=False)
print("== S1 eager batch 1: pairs resolved", int(S1.resolved.sum()),"/",len(S1))
print(R[(R.batch==1)&(R.runtime=="eager")][["family","x","y","mx","my","rel","p_holm","resolved"]].round(4).to_string())
print("\n== categories vs eager-b1, every setting")
C=R.groupby(["batch","runtime"]).category.value_counts().unstack(fill_value=0); print(C.to_string())
print("\n== H1b settings (batch 1, host overhead removed)")
for r in ["eager_graph","trt_fp32_graph","trt_fp16_graph","trt_int8a_graph","trt_int8b_graph","compile_ro"]:
    x=R[(R.batch==1)&(R.runtime==r)]; rev=(x.category=="strict reversal").sum(); el=x.category.isin(["emergent difference","lost advantage"]).sum()
    print(f"{r:16s} strict reversals {rev}  emergent/lost {el}  -> {'confirmed' if rev>=1 or el>=2 else ('refuted' if (x.category=='unchanged').all() else 'inconclusive')}")
    print("   ", "; ".join(f"{a.x}-{a.y}: {a.category} ({a.rel*100:+.1f}%)" for a in x.itertuples() if a.category!="unchanged"))
print("\n== strict reversals anywhere")
print(R[R.category=="strict reversal"][["batch","runtime","family","x","y","rel"]].assign(rel=lambda d:(d.rel*100).round(1)).to_string())
# H1a
P=pd.read_csv("stageP/profile.csv"); P=P[P.stage=="P"]
calls=P[(P.runtime=="eager")&(P.batch==1)].set_index("cfg").leaf_calls
nk=P[(P.runtime=="eager")&(P.batch==1)].set_index("cfg").n_kernel
lat=L[(L.runtime=="eager")&(L.batch==1)].groupby("cfg").median_ms.median()
print("\n== H1a Spearman eager-b1 latency vs leaf calls: rho=%.3f p=%.4f ; vs kernels: rho=%.3f" % (*stats.spearmanr(lat[calls.index],calls), stats.spearmanr(lat[nk.index],nk)[0]))
s640=[c for c in calls.index if c not in ("base960","base1024","base1120","p2_832","p2_960")]
print("   640-input subset: rho=%.3f" % stats.spearmanr(lat[s640],calls[s640])[0])
# H1c
m=L.groupby(["cfg","batch","runtime"]).median_ms.median()
for b in (1,8,32):
    print(f"\n== H1c batch {b}: base1120 vs base640 " + ", ".join(f"{r} {(m[('base1120',b,r)]/m[('base640',b,r)]-1)*100:+.1f}%" for r in ["eager","eager_graph","trt_fp16_graph","trt_fp32_graph","trt_int8a_graph"]))
# OLS
print("\n== OLS latency ~ N_kernel + GFLOPs (per runtime, batch)")
out=[]
for (b,r),g in P.groupby(["batch","runtime"]):
    if r=="eager_unfused": continue
    g=g.set_index("cfg"); y=m.xs((b,r),level=["batch","runtime"])[g.index].values
    X1=np.c_[np.ones(len(g)),g.n_kernel.values]; X2=np.c_[X1,[GF[c] for c in g.index]]
    def fit(X):
        beta,res,_,_=np.linalg.lstsq(X,y,rcond=None); yh=X@beta; r2=1-((y-yh)**2).sum()/((y-y.mean())**2).sum()
        n,k=X.shape; s2=((y-yh)**2).sum()/(n-k); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X))); return beta,se,r2
    b1,_,r2a=fit(X1); b2,se2,r2b=fit(X2); t=stats.t.ppf(0.975,len(g)-3)
    out.append(dict(batch=b,runtime=r,R2_kernels=r2a,R2_both=r2b,dR2_gflops=r2b-r2a,us_per_kernel=b2[1]*1000,
                    us_per_kernel_ci=f"[{(b2[1]-t*se2[1])*1000:.1f}, {(b2[1]+t*se2[1])*1000:.1f}]",ms_per_gflop=b2[2],
                    ms_per_gflop_ci=f"[{b2[2]-t*se2[2]:.3f}, {b2[2]+t*se2[2]:.3f}]"))
O=pd.DataFrame(out); O.to_csv("rq2_ols.csv",index=False); print(O.round(3).to_string())
# launch-dispatch effect: eager - eager_graph per kernel, batch 1
dd=((m.xs((1,"eager"),level=["batch","runtime"])-m.xs((1,"eager_graph"),level=["batch","runtime"]))/nk*1000)
print("\n== eager - eager+graph at batch 1, per executed kernel (us): median %.2f, range %.2f-%.2f" % (dd.median(),dd.min(),dd.max()))
