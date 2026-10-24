import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_rows",300); pd.set_option("display.max_columns",30)
A=pd.read_csv("sc_full/accuracy.csv"); C=pd.read_csv("sc_full/coco.csv")
M=A.merge(C[["run","path","split","coco_AP","AP_s","AP_m","AP_l"]],on=["run","path","split"])
M.to_csv("rq4_accuracy_runs.csv",index=False)
T=M[M.split=="test"].copy()
base=T[T.path=="eager_sq"].set_index("run")
for col in ("map5095","coco_AP","AP_s"):
    T["d_"+col]=(T[col]-T.run.map(base[col]))*100                     # points
    T["rel_"+col]=(T[col]/T.run.map(base[col])-1)*100                  # percent
S=T.groupby(["cfg","path"]).agg(n=("run","size"),map=("map5095","mean"),d=("d_map5095","mean"),d_sd=("d_map5095","std"),
                                 rel_map=("rel_coco_AP","mean"),rel_aps=("rel_AP_s","mean"),aps=("AP_s","mean"))
print("== test-dev mAP@0.5:0.95 change vs eager FP32 (points, seed mean ± SD)")
print(S["d"].unstack("path")[["ort_fp32","trt_fp32","trt_fp16","trt_int8a","trt_int8b"]].round(3).to_string())
f=S.xs("trt_fp16",level="path"); a=f.d.abs()
print(f"\n== H4a: |Δ| FP16 <=0.15 in {int((a<=0.15).sum())}/20, max {a.max():.3f} ({a.idxmax()}), >0.30: {int((a>0.30).sum())}")
for sch in ("trt_int8a","trt_int8b"):
    x=S.xs(sch,level="path"); loss_s=-x.rel_aps; loss_m=-x.rel_map; k=(loss_s>loss_m)
    print(f"\n== H4b {sch}: relative AP_small loss > relative AP loss in {int(k.sum())}/20  (median AP loss {loss_m.median():.2f}%, AP_small loss {loss_s.median():.2f}%)")
    print(pd.DataFrame({"AP_loss%":loss_m,"APs_loss%":loss_s,"mAP_d_pts":x.d}).round(2).sort_values("APs_loss%").to_string())
S.reset_index().to_csv("rq4_summary.csv",index=False)
# sanity
s=pd.read_csv("sc_full/sanity.csv"); print("\n== sanity (max over seeds)"); print(s.groupby("path")[["max_abs","mean_abs","max_abs_scores"]].max().round(4))
# H4c
L=pd.read_csv("sc_full/latency_C.csv")
P=L.pivot_table(index="cfg",columns=["unit","conf"],values="nms_share")
print("\n== H4c NMS share at batch 1"); print(P.round(3).to_string())
s640=[c for c in P.index if c in ("base640","p2_640","afterA","c1bn","c1pos","Awide","Bnarrow","Bplain","Bg4","seg1","seg4","safew288")]
for conf in (0.25,0.001):
    e=P[("eager",conf)]; t=P[("trt_fp16_graph",conf)]
    print(f"conf {conf}: TRT FP16 share >=0.25 in {int((t[s640]>=0.25).sum())}/{len(s640)} 640 YOLO11s-based; median eager {e.median():.3f} -> TRT {t.median():.3f}  ratio {t.median()/e.median():.2f}x")
print(L.groupby(["unit","conf"])[["fwd_ms","nms_ms"]].median().round(3))
