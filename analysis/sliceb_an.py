"""Slice B analysis: same pair test as rq1.py (two-stage bootstrap of medians, Holm within a setting,
resolved = Holm p < 0.05 and |diff| >= max(2 %, 0.05 ms), categories vs eager batch 1 on the same GPU),
restricted to the Slice B configurations and settings, for each GPU incl. the 4090 core data."""
import sys, json
import numpy as np, pandas as pd

GPUS = dict(a.split("=") for a in sys.argv[1:])          # name=latency.csv
B = 20000
PAIRS = [("D1", "p2_832", "base960"), ("D1", "p2_960", "base1120"), ("D1b", "c1pos", "base960"),
         ("D2", "c1pos", "y11m"), ("D2*", "c1pos", "y11l"), ("D3", "c1pos", "Bplain")]
CFGS = ["base640", "p2_640", "c1pos", "Bplain", "y11m", "y11l", "p2_832", "base960", "p2_960", "base1120"]
RTS = ["eager", "eager_graph", "trt_fp16_nograph", "trt_fp16_graph"]


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj


rows, meds, refs = [], {}, {}
for gname, path in GPUS.items():
    rng = np.random.default_rng(20261016)
    L = pd.read_csv(path)
    refs[gname] = L[L.runtime == "ref_eager"].median_ms.describe()[["count", "min", "50%", "max"]].round(3).to_dict()
    L = L[L.cfg.isin(CFGS) & L.batch.isin([1, 8]) & L.runtime.isin(RTS)]
    groups = {k: [g.median_ms.values for _, g in v.groupby("build")] for k, v in L.groupby(["cfg", "batch", "runtime"])}
    cache = {}

    def bs(k):
        if k not in cache:
            gs = groups[k]; nb = len(gs); out = np.empty(B); bi = rng.integers(0, nb, (B, nb))
            for j in range(B):
                out[j] = np.median(np.concatenate([gs[b][rng.integers(0, len(gs[b]), len(gs[b]))] for b in bi[j]]))
            cache[k] = out
        return cache[k]

    for k, gs in groups.items():
        meds[(gname,) + k] = float(np.median(np.concatenate(gs)))
    for b in (1, 8):
        for r in RTS:
            part = []
            for fam, x, y in PAIRS:
                kx, ky = (x, b, r), (y, b, r)
                mx, my = meds[(gname,) + kx], meds[(gname,) + ky]
                d = bs(kx) - bs(ky); diff = mx - my
                p = max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / B)
                part.append(dict(gpu=gname, batch=b, runtime=r, family=fam, x=x, y=y, mx=mx, my=my,
                                 ratio=mx / my - 1, p=p, practical=abs(diff) >= max(0.02 * min(mx, my), 0.05)))
            ph = holm(np.array([q["p"] for q in part]))
            for q, a in zip(part, ph):
                q["p_holm"] = a; q["resolved"] = (a < 0.05) and q["practical"]
                q["sign"] = int(np.sign(q["mx"] - q["my"])) if q["resolved"] else 0
            rows += part
R = pd.DataFrame(rows)
ref = R[(R.batch == 1) & (R.runtime == "eager")].set_index(["gpu", "x", "y"])


def cat(q):
    s = ref.loc[(q.gpu, q.x, q.y)]
    if s.resolved and q.resolved:
        return "unchanged" if s.sign == q.sign else "strict reversal"
    if s.resolved:
        return "lost advantage"
    if q.resolved:
        return "emergent difference"
    return "unchanged"


R["category"] = R.apply(cat, axis=1)
R.to_csv("sliceb_pairs.csv", index=False)
M = pd.Series(meds).rename_axis(["gpu", "cfg", "batch", "runtime"]).rename("median_ms").reset_index()
M.to_csv("sliceb_medians.csv", index=False)
print("drift reference (base640 eager b1, ms):", json.dumps(refs, indent=0))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("\n== pair ratios x/y - 1 (%), * = resolved; category vs eager b1 on that GPU")
R["cell"] = R.apply(lambda q: f"{q.ratio * 100:+.0f}{'*' if q.resolved else ''} {q.category[:6] if q.category != 'unchanged' else ''}".strip(), axis=1)
print(R.pivot_table(index=["family", "x", "y", "batch", "runtime"], columns="gpu", values="cell", aggfunc="first").to_string())
print("\n== categories per GPU")
print(R.groupby(["gpu"]).category.value_counts().unstack(fill_value=0).to_string())
# H1c analogue and host-overhead proxies
P = M.pivot_table(index=["cfg", "batch", "runtime"], columns="gpu", values="median_ms")
print("\n== base1120 / base640 - 1 (%)")
for b in (1, 8):
    for r in RTS:
        print(b, r, {g: round((P.loc[("base1120", b, r), g] / P.loc[("base640", b, r), g] - 1) * 100, 1) for g in GPUS})
print("\n== eager / eager_graph at batch 1 (host-overhead factor), and GPU speed ratio vs 4090 (trt_fp16_graph b8)")
for c in CFGS:
    print(f"{c:9s}", {g: round(P.loc[(c, 1, "eager"), g] / P.loc[(c, 1, "eager_graph"), g], 2) for g in GPUS},
          {g: round(P.loc[(c, 8, "trt_fp16_graph"), g] / P.loc[(c, 8, "trt_fp16_graph"), "4090"], 2) for g in GPUS})
print("\n== spearman of per-setting config ranking vs 4090")
from scipy.stats import spearmanr
for b in (1, 8):
    for r in RTS:
        print(b, r, {g: round(spearmanr([P.loc[(c, b, r), "4090"] for c in CFGS], [P.loc[(c, b, r), g] for c in CFGS])[0], 3) for g in GPUS if g != "4090"})
