import pandas as pd, numpy as np, itertools, pickle, os
from scipy import stats
rng=np.random.default_rng(20261016); B=4000
L=pd.read_csv("full_res/latency.csv"); L=L[L.runtime!="ref_eager"]
groups={k:[g.median_ms.values for _,g in v.groupby("build")] for k,v in L.groupby(["cfg","batch","runtime"])}
cache="h1d_boot.pkl"
if os.path.exists(cache): BS=pickle.load(open(cache,"rb"))
else:
    BS={}
    for k,gs in groups.items():
        nb=len(gs); out=np.empty(B)
        for j in range(B):
            bi=rng.integers(0,nb,nb); out[j]=np.median(np.concatenate([gs[b][rng.integers(0,len(gs[b]),len(gs[b]))] for b in bi]))
        BS[k]=out
    pickle.dump(BS,open(cache,"wb"))
med={k:np.median(np.concatenate(v)) for k,v in groups.items()}
A=pd.read_csv("rq4_accuracy_runs.csv"); A=A[A.split=="test"]
PATH={"eager":"eager_sq","eager_graph":"eager_sq","compile_default":"eager_sq","compile_ro":"eager_sq","ort_iobinding":"ort_fp32",
      "ort_with_copies":"ort_fp32","trt_fp32_graph":"trt_fp32","trt_fp16_graph":"trt_fp16","trt_fp16_nograph":"trt_fp16",
      "trt_int8a_graph":"trt_int8a","trt_int8b_graph":"trt_int8b"}
LAT={"base960b16":"base960"}                       # batch-16 baseline@960 shares the latency cell (deviation 1)
CFGS=sorted(A.cfg.unique())
def seeds(cfg,path):
    x=A[(A.cfg==cfg)&(A.path==path)].sort_values("seed"); return x.seed.values, x.map5095.values*100
def acc_cmp(x,y,path):                              # +1 x more accurate, -1 less, 0 not different
    sx,vx=seeds(x,path); sy,vy=seeds(y,path)
    if len(vx)<2 and len(vy)<2: return 0, np.nan     # two single-seed configurations: no test possible
    if len(vx)<2 or len(vy)<2:                       # one single seed: one-sample t-test of the multi-seed side
        single,multi,sgn=(vx[0],vy,1) if len(vx)<2 else (vy[0],vx,-1)
        p=stats.ttest_1samp(multi,single).pvalue
        return (sgn*int(np.sign(single-multi.mean())) if p<0.05 else 0), p
    common=sorted(set(sx)&set(sy))
    if len(common)==len(sx)==len(sy):              # paired-seed rule (same seeds)
        p=stats.ttest_rel(vx,vy).pvalue
    else:
        p=stats.ttest_ind(vx,vy,equal_var=False).pvalue
    return (int(np.sign(vx.mean()-vy.mean())) if p<0.05 else 0), p
def lat_cmp(x,y,b,r):                               # +1 x faster (resolved), -1 slower, 0 unresolved
    kx,ky=(LAT.get(x,x),b,r),(LAT.get(y,y),b,r)
    if kx==ky: return 0
    d=BS[kx]-BS[ky]; lo,hi=np.percentile(d,[2.5,97.5]); diff=med[kx]-med[ky]
    res=(lo>0 or hi<0) and abs(diff)>=max(0.02*min(med[kx],med[ky]),0.05)
    return (-int(np.sign(diff)) if res else 0)
rows=[]; fronts={}
for (b,r) in sorted({(k[1],k[2]) for k in groups}):
    path=PATH[r]; acc={}
    for x,y in itertools.permutations(CFGS,2): pass
    AC={}; 
    for x,y in itertools.combinations(CFGS,2):
        a,_=acc_cmp(x,y,path); AC[(x,y)]=a; AC[(y,x)]=-a
    LC={}
    for x,y in itertools.combinations(CFGS,2):
        l=lat_cmp(x,y,b,r); LC[(x,y)]=l; LC[(y,x)]=-l
    dom=lambda x,y: LC[(x,y)]>=0 and AC[(x,y)]>=0 and (LC[(x,y)]>0 or AC[(x,y)]>0)
    front=sorted(c for c in CFGS if not any(dom(o,c) for o in CFGS if o!=c))
    fronts[(b,r)]=front
F0=set(fronts[(1,"eager")])
print("eager batch 1 front:", sorted(F0))
out=[]
for k,f in fronts.items():
    f=set(f); out.append(dict(batch=k[0],runtime=k[1],size=len(f),entered=" ".join(sorted(f-F0)),left=" ".join(sorted(F0-f)),front=" ".join(sorted(f))))
O=pd.DataFrame(out); O.to_csv("h1d_fronts_v2.csv",index=False)
pd.set_option("display.width",300); pd.set_option("display.max_colwidth",120)
print(O.to_string())
print("settings with a changed front:", int(((O.entered!="")|(O.left!="")).sum()), "of", len(O))
# accuracy ranking for context
m=A[A.path=="eager_sq"].groupby("cfg").map5095.agg(["mean","std","count"]); print((m*[100,100,1]).round(2).sort_values("mean").to_string())

# who dominates base1120 where it leaves the front
for (b,r),f in fronts.items():
    if "base1120" in f: continue
    path=PATH[r]
    for o in CFGS:
        if o=="base1120": continue
        a,p=acc_cmp(o,"base1120",path); l=lat_cmp(o,"base1120",b,r)
        if l>=0 and a>=0 and (l>0 or a>0):
            print("dominates base1120:",b,r,o,"acc",a,round(p,3) if p==p else p,"lat",l)
