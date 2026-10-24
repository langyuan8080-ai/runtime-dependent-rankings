import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_rows",300); pd.set_option("display.max_columns",40)
a=pd.read_csv("extra_res/latency_A.csv"); a=a[(a.status=="ok")&(a["pass"].isin(["1","2"]))]
a["mps"]=a.mps.astype(str)
W=a.groupby(["cfg","imgsz","batch","runtime","mps"]).wall_median_ms.median().rename("t_wall")
p=pd.read_csv("extra_res/profile.csv"); p=p[(p.stage=="A")&(p.status=="ok")].copy(); p["mps"]=p.mps.astype(str)
T=p.groupby(["cfg","imgsz","batch","runtime","mps"]).t_dev_ms.median()
D=pd.concat([W,T],axis=1); D["h"]=1-D.t_dev_ms/D.t_wall
D.to_csv("rq3_cells.csv")
e=D.xs("eager",level="runtime")
print("== eager host share h by batch (rows: cfg, input, MPS%)")
print(e.h.unstack("batch").round(2).to_string())
def cross(g):
    g=g.dropna().sort_index()
    below=[b for b,v in g.items() if v<0.5]
    return min(below) if below else np.nan
C=e.h.groupby(level=["cfg","imgsz","mps"]).apply(lambda s: cross(s.droplevel([0,1,3]))).unstack("mps")[["100","50","25"]]
C["monotone_decrease"]=(C["100"]>=C["50"])&(C["50"]>=C["25"])&(C["25"]<C["100"])
C["non_increasing"]=(C["100"]>=C["50"])&(C["50"]>=C["25"])
print("\n== eager crossover batch (smallest batch with h < 0.5)"); print(C.to_string())
for s in (640,960,1280):
    x=C.xs(s,level="imgsz"); print(f"input {s}: monotone decrease in {int(x.monotone_decrease.sum())}/4 configurations")
x=C.groupby(level="cfg").monotone_decrease.any(); print("any input:", x.to_dict())
# slowdown factors per MPS level, device-bound cells
print("\n== latency ratio MPS50/MPS100 and MPS25/MPS100 (median over batch>=8 cells)")
L=a.groupby(["cfg","imgsz","batch","runtime","mps"]).median_ms.median().unstack("mps")
L["r50"]=L["50"]/L["100"]; L["r25"]=L["25"]/L["100"]
print(L[L.index.get_level_values("batch")>=8].groupby(level="runtime")[["r50","r25"]].median().round(2))
print(L.xs(1,level="batch").groupby(level="runtime")[["r50","r25"]].median().round(2))
