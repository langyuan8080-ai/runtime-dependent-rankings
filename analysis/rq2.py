import pandas as pd, numpy as np, json, glob
pd.set_option("display.width",250); pd.set_option("display.max_rows",200); pd.set_option("display.max_columns",30)
L=pd.read_csv("full_res/latency.csv"); L=L[L.runtime!="ref_eager"].copy()
# T_wall per cell: TRT -> build 1 (the engine profiled); compile -> median over 3 builds (profiling compiled anew)
L["use"]=~L.runtime.str.startswith("trt") | (L.build==1)
W=L[L.use].groupby(["cfg","batch","runtime"]).wall_median_ms.median().rename("t_wall")
P=pd.read_csv("stageP/profile.csv"); P=P[P.stage=="P"].set_index(["cfg","batch","runtime"])
D=P.join(W,how="left")
D["h"]=1-D.t_dev_ms/D.t_wall
D["exp_host_us_per_kernel"]=(D.t_wall-D.t_dev_ms)/D.n_kernel*1000
D.to_csv("rq2_cells.csv")
s_class=[c for c in D.index.get_level_values(0).unique() if c not in ("y11m","y11l")]
w20=lambda c: 2 if c=="base960" else 1          # base960 latency cell stands for two configurations (deviation 1)
b1=D.xs(1,level="batch")
print("== H2a host share, eager, batch 1 (YOLO11s-class)")
h=b1.xs("eager",level="runtime").h
print(h.round(3).sort_values().to_string()); print("min s-class", h[s_class].min().round(3), " all>=0.70:", (h[s_class]>=0.70).all(), " any<0.50:", (h[s_class]<0.5).any())
print("\n== H2b exposed host time per kernel (us), batch 1: eager vs compile_default")
e=b1.xs("eager",level="runtime").exp_host_us_per_kernel; c=b1.xs("compile_default",level="runtime").exp_host_us_per_kernel
t=pd.DataFrame({"eager":e,"compile_default":c,"n_k_eager":b1.xs("eager",level="runtime").n_kernel,"n_k_compile":b1.xs("compile_default",level="runtime").n_kernel})
print(t.round(2).to_string())
we=np.average(e,weights=[w20(i) for i in e.index]); wc=np.average(c,weights=[w20(i) for i in c.index])
print(f"mean eager {we:.2f} us, compile {wc:.2f} us, difference {abs(wc-we)/we*100:.1f}%  (confirmed <=15, refuted >30)")
print("\n== H2c device time at batch 1 with CUDA Graphs: reduction vs eager_graph")
g=b1.xs("eager_graph",level="runtime").t_dev_ms
r=pd.DataFrame({"eager_graph":g,"compile_ro":b1.xs("compile_ro",level="runtime").t_dev_ms,"trt_fp32":b1.xs("trt_fp32_graph",level="runtime").t_dev_ms})
r["red_compile%"]=(1-r.compile_ro/r.eager_graph)*100; r["red_trt32%"]=(1-r.trt_fp32/r.eager_graph)*100
print(r.round(3).to_string())
for col in ("red_compile%","red_trt32%"):
    v=np.repeat(r[col].values,[w20(i) for i in r.index])
    print(col, "median", round(np.median(v),1), "base640", round(r.loc["base640",col],1))
print("\n== H3b host share at batch 1 under CUDA Graph / TensorRT")
rts=["eager_graph","compile_ro","trt_fp32_graph","trt_fp16_graph","trt_int8a_graph","trt_int8b_graph"]
H=b1.h.unstack("runtime")[rts]
print(H.round(3).to_string())
for rt in rts:
    v=H[rt]; n20=sum(w20(i) for i in v.index if v[i]<0.30); 
    print(f"{rt:16s} <0.30: {n20}/20   max {v.max():.3f}  all<0.40: {(v<0.40).all()}  >=0.50: {sum(w20(i) for i in v.index if v[i]>=0.5)}")
print("\n== host share by batch (median over configurations)")
print(D.h.unstack("runtime").groupby(level="batch").median().round(3).T.to_string())
