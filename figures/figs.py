"""Figures of the paper (Springer double-column width 174 mm). Run by reproduce.sh from analysis/output/."""
import pickle, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

W = 174 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "pdf.fonttype": 42, "savefig.dpi": 600, "hatch.color": "#b9b8b2", "hatch.linewidth": 0.4})
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e1"
RT = ["eager", "eager_graph", "compile_default", "compile_ro", "ort_iobinding", "ort_with_copies", "trt_fp32_graph",
      "trt_fp16_graph", "trt_fp16_nograph", "trt_int8a_graph", "trt_int8b_graph"]
RTL = {"eager": "eager", "eager_graph": "eager + graph", "compile_default": "compile", "compile_ro": "compile RO",
       "ort_iobinding": "ORT I/O", "ort_with_copies": "ORT copies", "trt_fp32_graph": "TRT FP32", "trt_fp16_graph": "TRT FP16",
       "trt_fp16_nograph": "TRT FP16 ng", "trt_int8a_graph": "TRT INT8 a", "trt_int8b_graph": "TRT INT8 b"}
NAME = {"base640": "Baseline 640", "p2_640": "+P2 640", "afterA": "afterA", "c1bn": "c1bn", "c1pos": "c1pos", "Awide": "A-wide",
        "Bnarrow": "B-narrow", "Bplain": "B-plain", "Bg4": "B-g4", "seg1": "Conv g1+SE", "seg4": "Conv g4+SE",
        "safew288": "SAFEW-288", "y11m": "YOLO11m", "y11l": "YOLO11l", "base960": "Baseline 960", "base960b16": "Baseline 960 (b16)",
        "base1024": "Baseline 1024", "base1120": "Baseline 1120", "p2_832": "+P2 832", "p2_960": "+P2 960"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"figs/{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=600 if ext == "pdf" else 300)
    plt.close(fig)


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig. 1 study design
def fig1():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis("off")
    boxes = [
        (1, 21, 31, 17, "Core layer  (RQ1, RQ2)", "19 latency configurations\n11 runtime–precision cells\nbatch 1, 8, 32 · 3 builds\n627 cells · RTX 4090", BLUE),
        (34, 21, 31, 17, "Profiling layer  (RQ2)", "same grid in Nsight Systems\nseparate processes\nT_dev, N_kernel, host share h\n646 cells", BLUE),
        (67, 21, 32, 17, "Slice A  (RQ3)", "4 configurations · 640/960/1280\nbatch 1–32 · 3 runtimes\nMPS 100 / 50 / 25 % of SMs\n642 of 648 cells completed", AQUA),
        (1, 1.5, 48, 16.5, "Slice B  (RQ1, RQ3 external validity)", "10 configurations · eager, graph, TRT FP16\nbatch 1, 8 · RTX 3080 Ti and RTX 5090\nsame software image, other host CPUs", ORANGE),
        (51, 1.5, 48, 16.5, "Slice C  (RQ4, accuracy axis of RQ1)", "all 68 seeds · 6 precision paths\nVisDrone val + test-dev · 816 evaluations\nNMS share at batch 1 (200 images)", YELLOW),
    ]
    for x, y, w, h, title, body, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=col, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y + h - 4.2), w, 4.2, boxstyle="round,pad=0,rounding_size=1.2", fc=col, ec=col, lw=1.2, alpha=0.16))
        ax.text(x + 1.5, y + h - 2.1, title, va="center", ha="left", fontsize=7.5, weight="bold", color=INK)
        ax.text(x + 1.5, y + h - 5.6, body, va="top", ha="left", fontsize=6.6, color=INK2, linespacing=1.35)
    ax.text(50, 42.8, "Protocol frozen 16 Oct 2026 · 20 accuracy configurations from 68 training runs (VisDrone2019-DET)",
            ha="center", va="top", fontsize=7, color=INK2)
    save(fig, "ESM1_Fig_design")


# ------------------------------------------------------------------ Fig. 2 D2 heat map
def fig2():
    R = pd.read_csv("rq1_pairs.csv")
    D = R[R.family == "D2"].copy()
    D["ratio"] = (D.mx / D.my - 1) * 100
    pairs = [("c1pos", "y11m"), ("c1pos", "y11l"), ("safew288", "y11m"), ("safew288", "y11l")]
    cols = [(b, r) for b in (1, 8, 32) for r in RT]
    M = np.full((4, len(cols)), np.nan); RES = np.zeros_like(M, bool); SR = np.zeros_like(M, bool)
    for i, (x, y) in enumerate(pairs):
        for j, (b, r) in enumerate(cols):
            q = D[(D.x == x) & (D.y == y) & (D.batch == b) & (D.runtime == r)].iloc[0]
            M[i, j] = q.ratio; RES[i, j] = q.resolved; SR[i, j] = q.category == "strict reversal"
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a98", "#b42d2d"])
    lim = 95
    fig, ax = plt.subplots(figsize=(W, 2.15))
    ax.imshow(np.where(RES, M, np.nan), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    # unresolved cells: neutral grey with hatch
    for i, j in zip(*np.where(~RES)):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#f7f6f3", ec="none", hatch="////", lw=0))
    for i, j in itertools.product(range(4), range(len(cols))):
        v = M[i, j]
        ax.text(j, i, f"{v:.0f}".replace("-", "\u2212"), ha="center", va="center", fontsize=5.2,
                color="white" if RES[i, j] and abs(v) > 55 else INK)
        if SR[i, j]:
            ax.add_patch(plt.Rectangle((j - .46, i - .46), .92, .92, fc="none", ec=INK, lw=1.1))
    for k in (11, 22):
        ax.axvline(k - .5, color="white", lw=3)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([RTL[r] for b, r in cols], rotation=90, fontsize=5.6)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{NAME[x]} vs {NAME[y]}" for x, y in pairs], fontsize=6.5)
    for k, b in enumerate((1, 8, 32)):
        ax.text(k * 11 + 5, -0.85, f"batch {b}", ha="center", va="bottom", fontsize=7, weight="bold", color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # legend
    cax = fig.add_axes([0.80, -0.36, 0.17, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -lim, lim), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-80, -40, 0, 40, 80]); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
    cb.set_label("latency of module config / scaled model − 1 (%)", fontsize=6.3)
    fig.text(0.0, -0.31, "Blue: the module configuration is faster; red: slower.  Hatched: difference not resolved.  "
             "Black frame: strict reversal relative to eager batch 1.", fontsize=6.3, color=INK2)
    save(fig, "Fig1_D2_heatmap")


# ------------------------------------------------------------------ Fig. 3 Pareto fronts
def fig3():
    BS = pickle.load(open("h1d_boot.pkl", "rb"))
    L = pd.read_csv("full_res/latency.csv"); L = L[L.runtime != "ref_eager"]
    med = L.groupby(["cfg", "batch", "runtime"]).median_ms.median()
    A = pd.read_csv("rq4_accuracy_runs.csv"); A = A[A.split == "test"]
    F = pd.read_csv("h1d_fronts_v2.csv")
    panels = [(1, "eager", "eager_sq", "Eager, batch 1"), (1, "trt_fp16_graph", "trt_fp16", "TensorRT FP16 + graph, batch 1"),
              (32, "trt_fp16_graph", "trt_fp16", "TensorRT FP16 + graph, batch 32")]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.6))
    for ax, (b, r, path, title) in zip(axs, panels):
        front = set(F[(F.batch == b) & (F.runtime == r)].front.iloc[0].split())
        acc = A[A.path == path].groupby("cfg").map5095.agg(["mean", "std", "count"])
        pts = []
        for cfg in acc.index:
            lc = "base960" if cfg == "base960b16" else cfg
            m = med[(lc, b, r)]; lo, hi = np.percentile(BS[(lc, b, r)], [2.5, 97.5])
            per_img = 1 if b == 1 else b
            pts.append((cfg, m / per_img, lo / per_img, hi / per_img, acc.loc[cfg, "mean"] * 100,
                        (acc.loc[cfg, "std"] * 100 if acc.loc[cfg, "count"] > 1 else 0), int(acc.loc[cfg, "count"])))
        for cfg, m, lo, hi, a, sd, n in pts:
            on = cfg in front
            ax.errorbar(m, a, xerr=[[m - lo], [hi - m]], yerr=sd, fmt="none", ecolor=BLUE if on else GREY, elinewidth=0.6, capsize=0)
            ax.plot(m, a, "o", ms=4.2 if on else 3.4, mfc=BLUE if on else "white", mec=BLUE if on else GREY, mew=0.9)
        fr = sorted([p for p in pts if p[0] in front], key=lambda p: p[1])
        ax.plot([p[1] for p in fr], [p[4] for p in fr], color=BLUE, lw=0.9, zorder=0)
        OFF = {"c1pos": (14, 14, "left"), "y11m": (14, -14, "left"), "y11l": (-14, 14, "right"), "Bplain": (-6, 5, "right"),
               "Bg4": (5, -3, "left"), "base960": (5, -5, "left"), "base1120": (5, 5, "left")}
        for cfg, m, lo, hi, a, sd, n in pts:
            if cfg == "base960b16":
                continue
            if cfg in front or cfg in ("c1pos", "y11l", "y11m"):
                dx, dy, ha = OFF.get(cfg, (4, -1, "left"))
                lab = "Baseline 960 (both groups)" if cfg == "base960" else NAME[cfg] + (" (1 seed)" if n == 1 else "")
                ax.annotate(lab, (m, a), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            fontsize=5.4, color=BLUE if cfg in front else INK2, va="center",
                            arrowprops=None if cfg in front else dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=2))
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.set_xlabel("latency per image (ms)" if b > 1 else "latency (ms)")
        clean(ax, "both")
    axs[0].set_ylabel("test-dev mAP@0.5:0.95 (%)")
    fig.text(0.0, -0.06, "Filled: on the Pareto front. Error bars: 95% bootstrap interval of the latency median; seed SD of mAP "
             "(YOLO11m and YOLO11l have one seed). Batch-32 latency divided by 32.", fontsize=6.3, color=INK2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig2_pareto")


# ------------------------------------------------------------------ Fig. 4 host share and NMS share
def fig4():
    C = pd.read_csv("rq2_cells.csv"); C = C[C.runtime != "eager_unfused"]
    S = pd.read_csv("sc_full/latency_C.csv"); S = S[S.conf == 0.25]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
    GRAPH = ["eager_graph", "compile_ro", "trt_fp32_graph", "trt_fp16_graph", "trt_int8a_graph", "trt_int8b_graph"]
    C = C.assign(rt2=C.runtime.where(~C.runtime.isin(GRAPH), "graph6"))
    series = [("eager", "eager", BLUE), ("compile_default", "torch.compile (default)", ORANGE),
              ("ort_with_copies", "ORT with copies", "#7a5cc7"), ("ort_iobinding", "ORT I/O binding", AQUA),
              ("trt_fp16_nograph", "TRT FP16, no graph", YELLOW), ("graph6", "six CUDA Graph cells (pooled)", GREY)]
    xs = [1, 8, 32]
    for rt, lab, col in series:
        g = C[C.rt2 == rt].groupby("batch").h
        md, lo, hi = g.median(), g.min(), g.max()
        a1.fill_between(range(3), [lo[b] for b in xs], [hi[b] for b in xs], color=col, alpha=0.12, lw=0)
        a1.plot(range(3), [md[b] for b in xs], "-o", color=col, lw=1.6, ms=3.6, label=lab)
        a1.text(-0.09, md[1], f"{md[1]:.2f}", fontsize=5.6, color=col if col != GREY else INK2, va="center", ha="right")
    a1.axhline(0.5, xmin=0.17, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a1.text(2.0, 0.485, "host-bound above the line", fontsize=5.8, color=INK2, va="top", ha="right")
    a1.set_xticks(range(3)); a1.set_xticklabels(["1", "8", "32"]); a1.set_xlabel("batch size")
    a1.set_xlim(-0.45, 2.1)
    a1.set_ylabel("host share h = 1 − T_dev / T_wall"); a1.set_ylim(-0.07, 1.02)
    a1.set_title("(a) Host share (median, band = range over 19 configurations)", loc="left", fontsize=7.2)
    a1.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.03, 1.03), handlelength=1.3, labelspacing=0.25, fontsize=5.9)
    clean(a1)
    P = S.pivot_table(index="cfg", columns="unit", values="nms_share")
    P = P.sort_values("trt_fp16_graph")
    y = np.arange(len(P))
    for i, (cfg, row) in enumerate(P.iterrows()):
        a2.plot([row.eager * 100, row.trt_fp16_graph * 100], [i, i], color=GRID, lw=1.6, zorder=0)
    a2.plot(P.eager * 100, y, "o", color=GREY, ms=3.6, label="eager FP32")
    a2.plot(P.trt_fp16_graph * 100, y, "o", color=BLUE, ms=3.8, label="TensorRT FP16 + graph")
    a2.set_yticks(y); a2.set_yticklabels([NAME[c] for c in P.index], fontsize=5.6)
    a2.axvline(25, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a2.text(25.6, -0.75, "25% criterion (H4c)", fontsize=5.8, color=INK2, va="top")
    top = len(P) - 1
    a2.annotate("eager FP32", (P.eager.iloc[-1] * 100, top), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6, color=INK2)
    a2.annotate("TensorRT FP16 + graph", (P.trt_fp16_graph.iloc[-1] * 100, top), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6, color=BLUE)
    a2.set_xlim(0, 60); a2.set_xlabel("NMS share of forward-plus-NMS latency (%)")
    a2.set_title("(b) NMS share, batch 1, conf 0.25", loc="left", fontsize=7.2)
    a2.set_ylim(-1.6, len(P) + 0.5)
    clean(a2, "x")
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig3_host_nms")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("done")
