#!/usr/bin/env python3
"""
extra.py — Paper 2, second batch on the same machine as the core layer (pre-registration rules 10-13, Slice A).

  python extra.py run   --out DIR [--smoke]     orchestrator (resumable; run in background)
  python extra.py watch DIR                      progress bars + ETA (Ctrl+C leaves the job running)
  python extra.py status DIR                     one-shot summary

Stage P (profiling, rules 10-12; no MPS): all 19 configurations x batch {1, 8, 32} x 11 runtime cells, plus eager
  without Conv-BN folding at batch 1. One nsys process per cell (never a timing process): 20 NVTX-marked forward
  passes -> executed kernels, kernel-launch API calls, graph launches, memcpy/memset, device time T_dev (union of
  kernel intervals) per pass, per-name kernel counts; leaf-module calls for eager.
Stage A (Slice A, rule 13): base640, p2_640, c1pos, Bplain x input {640, 960, 1280} x batch {1,2,4,8,16,32} x
  {eager, eager + CUDA Graph, TensorRT FP16 graph (3 builds)} x MPS {100, 50, 25} %. Latency as in the core layer
  (CUDA events, 50 + 300 passes, 3 + 2 repetitions in two passes, reference cell per block); every cell is also
  profiled with nsys for T_dev. MPS-100 % vs no MPS is compared once, at the start of the stage.
Requires core.py (frozen core-layer script) in the same directory.
"""
import argparse, collections, csv, json, os, random, re, shutil, sqlite3, subprocess, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                                                     # noqa: E402
from core import CFG, PV_PY, QV_PY, jdump, jload, cdir, bar, hms, THREADS, SEED  # noqa: E402

HERE = os.path.abspath(__file__)
FULL = os.environ.get("FULL_DIR", "/workspace/paper2_core/full")      # core-layer run (calibrated INT8 graphs)
NSYS = os.environ.get("NSYS") or shutil.which("nsys") or next(
    (p for p in sorted(__import__("glob").glob("/usr/local/cuda*/bin/nsys")) + sorted(
        __import__("glob").glob("/usr/local/cuda*/nsight-systems*/bin/nsys")) + sorted(
        __import__("glob").glob("/opt/nvidia/nsight-systems*/bin/nsys")) if os.path.exists(p)), "nsys")
N_PASSES = 20
P_RUNTIMES = ["eager", "eager_graph", "compile_default", "compile_ro", "ort_iobinding", "ort_with_copies",
              "trt_fp32_graph", "trt_fp16_graph", "trt_fp16_nograph", "trt_int8a_graph", "trt_int8b_graph"]
A_CFGS = ["base640", "p2_640", "c1pos", "Bplain"]
A_SIZES = [640, 960, 1280]
A_BATCHES = [1, 2, 4, 8, 16, 32]
A_MPS = [100, 50, 25]
A_BUILDS = 3
A_RUNTIMES = ["eager", "eager_graph", "trt_fp16_graph"]
MIN_FREE_GB = 4
DEFAULT_S = {"pprep": 90, "prebatch": 60, "build": 150, "prof": 30, "clean": 1, "mps": 3, "check": 60,
             "aprep": 120, "ref": 8, "atime_torch": 120, "atime_trt": 120}
TIMEOUT = {"build": 3600, "prof": 3600, "atime": 14400}
A_KEYS = ["ts", "cid", "cfg", "imgsz", "batch", "runtime", "mps", "build", "pass", "rep", "median_ms", "mean_ms",
          "sd_ms", "p95_ms", "n", "wall_median_ms", "status", "block"]
P_KEYS = ["ts", "stage", "cid", "cfg", "imgsz", "batch", "runtime", "mps", "passes", "n_kernel", "n_launch_api",
          "n_graph_launch", "n_memcpy", "n_memset", "t_dev_ms", "t_dev_p10_ms", "t_dev_p90_ms", "t_kernel_sum_ms",
          "t_span_ms", "t_range_ms", "leaf_calls", "status", "note"]


# ----------------------------------------------------------------------------- small helpers
def write_rows(path, keys, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def resolve(cid):
    """core configuration name -> (cfg, run, input); Slice-A id A_<cfg>_s<input> -> same with that input."""
    if cid.startswith("A_"):
        cfg, s = cid[2:].rsplit("_s", 1)
        return cfg, CFG[cfg][1], int(s)
    return cid, CFG[cid][1], CFG[cid][2]


def load_model(run_name, fuse=True):
    if fuse:
        return core.load_model(run_name)
    import torch.nn as nn
    from ultralytics import YOLO
    m = YOLO(os.path.join(core.RUNS, run_name, "weights", "best.pt")).model.float().eval()
    for mod in m.modules():
        if type(mod).__name__ == "SAFE":
            if not hasattr(mod, "sa"):
                mod.sa = nn.Identity()
            for a, v in (("use_spatial", False), ("gamma_per_channel", False), ("use_multiscale", True),
                         ("use_attention", True), ("use_gamma", True)):
                if not hasattr(mod, a):
                    setattr(mod, a, v)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.cuda()


def is_oom(ex):
    t = f"{type(ex).__name__} {ex}".lower()
    return "outofmemory" in t.replace(" ", "") or "out of memory" in t or "cudnn_status_alloc_failed" in t


# ----------------------------------------------------------------------------- runtime units (same set-up as core)
def torch_units(run_name, x, fuse=True):
    """eager and eager + CUDA Graph on one model instance, set up exactly as in the core layer."""
    import torch
    m = load_model(run_name, fuse=fuse)
    with torch.no_grad():
        m(x)                                        # Detect caches anchors for this shape
        sx = x.clone()
        st2 = torch.cuda.Stream()
        st2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st2):
            for _ in range(5):
                m(sx)
        torch.cuda.current_stream().wait_stream(st2)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m(sx)
    return m, {"eager": lambda: m(x), "eager_graph": lambda: (sx.copy_(x), g.replay())}, (g, sx)


def make_unit(out, cid, b, runtime, k=1):
    """Returns (fn, stream, keep). fn() runs one forward pass exactly as the core-layer timing did."""
    import torch
    torch.set_num_threads(THREADS)
    torch.backends.cudnn.benchmark = False
    cfg, run_name, s = resolve(cid)
    x = torch.randn(b, 3, s, s, device="cuda")
    if runtime in ("eager", "eager_unfused"):
        m = load_model(run_name, fuse=(runtime == "eager"))
        with torch.no_grad():
            m(x)
        return (lambda: m(x)), None, m
    if runtime == "eager_graph":
        m, units, keep = torch_units(run_name, x)
        return units["eager_graph"], None, (m, keep)
    if runtime.startswith("compile_"):
        mode = "default" if runtime == "compile_default" else "reduce-overhead"
        m = load_model(run_name)
        with torch.no_grad():
            m(x)
            torch._dynamo.reset()
            cm = torch.compile(m, mode=mode, dynamic=False)
            cm(x)
            torch.cuda.synchronize()
        return (lambda: cm(x)), None, None
    if runtime.startswith("ort_"):
        import numpy as np, onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(os.path.join(cdir(out, cid), "onnx", f"fp32_b{b}.onnx"), so,
                                    providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError("ORT fell back to CPU")
        oshape = [int(v) for v in sess.get_outputs()[0].shape]
        xin, yout = torch.empty_like(x), torch.empty(oshape, device="cuda")
        io = sess.io_binding()
        io.bind_input("images", "cuda", 0, np.float32, list(x.shape), xin.data_ptr())
        io.bind_output("out", "cuda", 0, np.float32, oshape, yout.data_ptr())
        xn = x.cpu().numpy()
        if runtime == "ort_iobinding":
            return (lambda: (xin.copy_(x), torch.cuda.synchronize(), sess.run_with_iobinding(io))), None, None
        return (lambda: sess.run(None, {"images": xn})), None, None
    if runtime.startswith("trt_"):
        _, prec, mode = runtime.split("_")
        trt = core.import_trt()
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        TDT = {trt.float32: torch.float32, trt.float16: torch.float16}
        path = os.path.join(cdir(out, cid), "engines", f"{prec}_b{b}_k{k}.plan")
        with open(path, "rb") as f:
            eng = rt.deserialize_cuda_engine(f.read())
        ctx = eng.create_execution_context()
        bufs = {}
        for i in range(eng.num_io_tensors):
            t = eng.get_tensor_name(i)
            bufs[t] = torch.zeros(tuple(eng.get_tensor_shape(t)), dtype=TDT[eng.get_tensor_dtype(t)], device="cuda")
            ctx.set_tensor_address(t, bufs[t].data_ptr())
        stream = torch.cuda.Stream()
        xsrc, inb = x.to(bufs["images"].dtype), bufs["images"]
        with torch.cuda.stream(stream):
            inb.copy_(xsrc)
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            if mode == "nograph":
                return (lambda: (inb.copy_(xsrc), ctx.execute_async_v3(stream.cuda_stream))), stream, (eng, ctx, bufs)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                ctx.execute_async_v3(stream.cuda_stream)
        return (lambda: (inb.copy_(xsrc), g.replay())), stream, (eng, ctx, bufs, g)
    raise ValueError(runtime)


def leaf_calls(m, fn):
    import torch
    n = [0]
    hs = [mod.register_forward_hook(lambda *a: n.__setitem__(0, n[0] + 1))
          for mod in m.modules() if not list(mod.children())]
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    for h in hs:
        h.remove()
    return n[0]


# ----------------------------------------------------------------------------- workers
def w_pprep(out, cfg):
    """Re-export FP32/FP16 graphs (deterministic; checked against the core-layer checksums) and bring the
    calibrated batch-1 INT8 graphs from the core run."""
    import hashlib
    core.w_prep(out, cfg)
    od, src = os.path.join(cdir(out, cfg), "onnx"), os.path.join(cdir(FULL, cfg), "onnx")
    q = jload(os.path.join(cdir(FULL, cfg), "quant.json"), {}) or {}
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    chk = {}
    for k, v in (q.get("fp32_md5") or {}).items():
        got = md5(os.path.join(od, k + ".onnx"))
        chk[k] = got == v
    jdump({"fp32_md5_match_core": chk}, os.path.join(cdir(out, cfg), "export_check.json"))
    if not all(chk.values()):
        print("WARNING: re-export differs from the core-layer export:", chk, flush=True)
    for p in ("int8a", "int8b"):
        shutil.copyfile(os.path.join(src, f"{p}_b1.onnx"), os.path.join(od, f"{p}_b1.onnx"))
    core.env_record(out, cfg, "P")
    print("PPREP_OK", cfg, chk)


def w_prebatch(out, cfg):
    import torch  # noqa: F401  loads the CUDA 13 libraries ONNX Runtime's CUDA provider needs (as in core timing)
    import numpy as np
    od = os.path.join(cdir(out, cfg), "onnx")
    s = CFG[cfg][2]
    x = np.random.default_rng(SEED).random((32, 3, s, s), dtype=np.float32)
    rec = {}
    for b in (8, 32):
        for p in ("int8a", "int8b"):
            dst = os.path.join(od, f"{p}_b{b}.onnx")
            r = core.rebatch(os.path.join(od, f"{p}_b1.onnx"), os.path.join(od, "fp32_b1.onnx"),
                             os.path.join(od, f"fp32_b{b}.onnx"), dst + ".tmp.onnx")
            c = core.check_batch(os.path.join(od, f"{p}_b1.onnx"), dst + ".tmp.onnx", x[:b], 2)
            if c["median"] > 1e-3 or c["p99"] > 5e-2:
                raise RuntimeError(f"{cfg} {p} b{b}: rebatched graph does not reproduce batch 1: {c}")
            os.replace(dst + ".tmp.onnx", dst)
            rec[f"{p}_b{b}"] = {"constants": r["batch_dependent_constants"], "vs_b1": c}
    jdump(rec, os.path.join(cdir(out, cfg), "rebatch.json"))
    print("PREBATCH_OK", cfg)


def w_aprep(out, cid):
    """Slice A: static FP16 ONNX at input s for every batch size (weights of the configuration's trained run)."""
    import copy, torch
    cfg, run_name, s = resolve(cid)
    od = os.path.join(cdir(out, cid), "onnx")
    os.makedirs(od, exist_ok=True)
    m16 = core.wrap(copy.deepcopy(load_model(run_name)).half())
    for b in A_BATCHES:
        p = os.path.join(od, f"fp16_b{b}.onnx")
        if os.path.exists(p):
            continue
        x = torch.zeros(b, 3, s, s, device="cuda", dtype=torch.half)
        with torch.no_grad():
            m16(x)
        torch.onnx.export(m16, (x,), p + ".tmp", opset_version=17, input_names=["images"], output_names=["out"],
                          dynamo=False)
        os.replace(p + ".tmp", p)
        del x
        torch.cuda.empty_cache()
    print("APREP_OK", cid)


def w_prof(out, cid, b, runtime, tag):
    """Runs under nsys: warm-up, then N_PASSES NVTX-marked forward passes, each followed by a synchronization."""
    import torch
    side = {"cid": cid, "batch": b, "runtime": runtime}
    try:
        fn, stream, keep = make_unit(out, cid, b, runtime)
        ctx = torch.cuda.stream(stream) if stream is not None else torch.no_grad()
        with torch.no_grad(), ctx:
            for _ in range(core.WARM):
                fn()
            torch.cuda.synchronize()
            if runtime in ("eager", "eager_unfused"):
                side["leaf_calls"] = leaf_calls(keep, fn)
            for i in range(N_PASSES):
                torch.cuda.nvtx.range_push(f"pass{i:02d}")
                fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        side["status"] = "ok"
    except FileNotFoundError as e:
        side["status"] = f"missing: {os.path.basename(str(e.filename))}"
    except Exception as ex:
        traceback.print_exc()
        if not is_oom(ex):
            raise
        side["status"] = "oom"
    jdump(side, os.path.join(out, "prof", tag + ".json"))
    print("PROF_DONE", side)


def w_atime(out, cid, b, fam, ps):
    import torch
    torch.set_num_threads(THREADS)
    torch.backends.cudnn.benchmark = False
    cfg, run_name, s = resolve(cid)
    mps = os.environ.get("EXTRA_MPS", "100")
    rng = random.Random(f"{SEED}-{cid}-{b}-{fam}-{ps}-{mps}")
    reps = core.REPS_P1 if ps in ("1", "check") else core.REPS_P2
    rows = []

    def R(runtime, build, rep, r, status="ok"):
        d = {"ts": time.strftime("%F %T"), "cid": cid, "cfg": cfg, "imgsz": s, "batch": b, "runtime": runtime,
             "mps": mps, "build": build, "pass": ps, "rep": rep, "status": status, "block": cid}
        d.update(r)
        return d
    try:
        if fam == "ref":
            core.env_record(out, cid, f"{ps}@mps{mps}")
            m = core.load_model(CFG["base640"][1])
            x1 = torch.randn(1, 3, 640, 640, device="cuda")
            with torch.no_grad():
                m(x1)
                r = core.time_fn(lambda: m(x1))
            rows.append(dict(R("ref_eager", 1, 1, r), cid="ref", cfg="base640", imgsz=640, batch=1, **{"pass": f"ref_{ps}"}))
        elif fam == "torch":
            x = torch.randn(b, 3, s, s, device="cuda")
            m, units, keep = torch_units(run_name, x)
            with torch.no_grad():
                for rep in range(1, reps + 1):
                    for u in rng.sample(list(units), len(units)):
                        rows.append(R(u, 1, rep, core.time_fn(units[u])))
        elif fam == "trt":
            edir = os.path.join(cdir(out, cid), "engines")
            ks = sorted(int(f.split("_k")[1][:-5]) for f in os.listdir(edir) if f.startswith(f"fp16_b{b}_k"))
            if not ks:
                rows.append(R("trt_fp16_graph", 0, 0, {}, status="no_engine"))
            for k in rng.sample(ks, len(ks)):
                try:
                    fn, stream, keep = make_unit(out, cid, b, "trt_fp16_graph", k=k)
                    for rep in range(1, reps + 1):
                        rows.append(R("trt_fp16_graph", k, rep, core.time_fn(fn, stream=stream)))
                    del fn, keep
                except Exception as ex:                     # recorded, not hidden: status column + log
                    traceback.print_exc()
                    rows.append(R("trt_fp16_graph", k, 0, {}, status="oom" if is_oom(ex) else f"error: {type(ex).__name__}"))
                torch.cuda.empty_cache()
    except Exception as ex:
        traceback.print_exc()
        if not is_oom(ex):
            raise
        rows.append(R("eager" if fam == "torch" else "trt_fp16_graph", 0, 0, {}, status="oom"))
    write_rows(os.path.join(out, "latency_A.csv"), A_KEYS, rows)
    print("ATIME_OK", cid, b, fam, ps, mps, len(rows), "rows")


# ----------------------------------------------------------------------------- nsys parsing (orchestrator side)
def parse_nsys(db):
    con = sqlite3.connect(db)
    tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    cols = lambda t: {r[1] for r in con.execute(f"pragma table_info({t})")}
    S = dict(con.execute("select id, value from StringIds")) if "StringIds" in tabs else {}
    name = lambda v: S.get(v, str(v)) if isinstance(v, int) else (v or "")
    # NVTX pass ranges (host time)
    nv = []
    if "NVTX_EVENTS" in tabs:
        c = cols("NVTX_EVENTS")
        tcol = "coalesce(text, '')" if "text" in c else "''"
        idcol = "textId" if "textId" in c else "NULL"
        for st, en, tx, tid in con.execute(f"select start, end, {tcol}, {idcol} from NVTX_EVENTS where end is not null"):
            label = tx or name(tid)
            if re.fullmatch(r"pass\d+", label or ""):
                nv.append((st, en))
    nv.sort()
    # API calls (runtime and driver) by correlation id
    api = []
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t in tabs:
            api += [(st, cid, name(nid)) for st, cid, nid in con.execute(f"select start, correlationId, nameId from {t}")]

    def act(t, extra=""):
        if t not in tabs:
            return []
        nc = "shortName" if "shortName" in cols(t) else ("demangledName" if "demangledName" in cols(t) else "NULL")
        q = f"select start, end, correlationId, {nc if t.endswith('KERNEL') else 'NULL'} from {t}"
        return list(con.execute(q))
    ker, mcp, mst = act("CUPTI_ACTIVITY_KIND_KERNEL"), act("CUPTI_ACTIVITY_KIND_MEMCPY"), act("CUPTI_ACTIVITY_KIND_MEMSET")
    con.close()
    rows, names = [], None
    for i, (ps, pe) in enumerate(nv):
        a = [(cid, n) for st, cid, n in api if ps <= st <= pe]
        corr = {cid for cid, _ in a}
        k = [(s, e, n) for s, e, c, n in ker if c in corr]
        iv = sorted((s, e) for s, e, _ in k)
        union, cur_s, cur_e = 0, None, None
        for s, e in iv:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    union += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            union += cur_e - cur_s
        rows.append({"n_kernel": len(k),
                     "n_launch_api": sum(1 for _, n in a if re.search(r"Launch(Kernel|CooperativeKernel)", n)),
                     "n_graph_launch": sum(1 for _, n in a if "GraphLaunch" in n),
                     "n_memcpy": sum(1 for s, e, c, _ in mcp if c in corr),
                     "n_memset": sum(1 for s, e, c, _ in mst if c in corr),
                     "t_dev_ms": union / 1e6, "t_kernel_sum_ms": sum(e - s for s, e in iv) / 1e6,
                     "t_span_ms": (max(e for _, e in iv) - min(s for s, _ in iv)) / 1e6 if iv else 0.0,
                     "t_range_ms": (pe - ps) / 1e6})
        if i == len(nv) // 2:
            names = collections.Counter(name(n) for _, _, n in k)
    return rows, names


def summarize(rows):
    import statistics as st
    if not rows:
        return {"passes": 0, "status": "no_passes"}
    use = rows[1:] if len(rows) > 2 else rows                    # first marked pass dropped (cold profiler buffers)
    med = lambda k: st.median(r[k] for r in use)
    td = sorted(r["t_dev_ms"] for r in use)
    out = {k: round(med(k), 6) if k.startswith("t_") else med(k) for k in
           ("n_kernel", "n_launch_api", "n_graph_launch", "n_memcpy", "n_memset", "t_dev_ms", "t_kernel_sum_ms",
            "t_span_ms", "t_range_ms")}
    out.update({"passes": len(use), "t_dev_p10_ms": round(td[int(0.1 * (len(td) - 1))], 6),
                "t_dev_p90_ms": round(td[int(0.9 * (len(td) - 1))], 6),
                "status": "ok" if out["n_kernel"] > 0 else "no_kernels"})
    return out


# ----------------------------------------------------------------------------- plan
def make_plan(smoke=False):
    rng = random.Random(SEED + 2)
    plan = []
    cfgs = ["base640"] if smoke else rng.sample([c[0] for c in core.CONFIGS], len(core.CONFIGS))
    for cfg in cfgs:                                                      # ---- stage P
        plan += [("pprep", cfg), ("prebatch", cfg)]
        bl = [("build", cfg, p, b, 1) for p in core.PRECS for b in core.BATCHES]
        if smoke:
            bl = [("build", cfg, p, 1, 1) for p in ("fp16", "int8a")]
        rng.shuffle(bl)
        plan += bl
        pr = [("prof", "P", cfg, b, r, "off") for b in core.BATCHES for r in P_RUNTIMES] + [("prof", "P", cfg, 1, "eager_unfused", "off")]
        if smoke:
            pr = [("prof", "P", cfg, 1, r, "off") for r in ("eager", "eager_unfused", "eager_graph", "compile_ro",
                                                           "ort_iobinding", "trt_fp16_graph", "trt_int8a_graph")]
        rng.shuffle(pr)
        plan += pr + [("clean", cfg)]
    # ---- stage A
    plan += [("check", "off"), ("mps", "on"), ("check", "100")]
    blocks = [f"A_{c}_s{s}" for c in A_CFGS for s in A_SIZES]
    sizes, batches, mpsl, nb = A_SIZES, A_BATCHES, A_MPS, A_BUILDS
    if smoke:
        blocks, batches, mpsl, nb = ["A_base640_s640"], [1, 32], [100, 25], 1
    rng.shuffle(blocks)
    for cid in blocks:
        plan.append(("aprep", cid))
        bl = [("build", cid, "fp16", b, k) for b in batches for k in range(1, nb + 1)]
        rng.shuffle(bl)
        plan += bl + [("ref", cid, "start")]
        cells = [(cid, b, fam, mp) for mp in mpsl for b in batches for fam in ("torch", "trt")]
        p1 = [("atime",) + c[:3] + ("1", c[3]) for c in cells]
        p2 = [("atime",) + c[:3] + ("2", c[3]) for c in cells]
        rng.shuffle(p1)
        rng.shuffle(p2)
        pr = [("prof", "A", cid, b, r, str(mp)) for mp in mpsl for b in batches for r in A_RUNTIMES]
        rng.shuffle(pr)
        plan += p1 + [("ref", cid, "mid")] + p2 + [("ref", cid, "end")] + pr + [("clean", cid)]
    plan.append(("mps", "off"))
    return [list(map(str, s)) for s in plan]


def sid(s):
    return "|".join(s)


def stype(s):
    if s[0] == "atime":
        return "atime_" + s[3]
    return s[0]


def need_mps(plan, i):
    """MPS daemon state a step needs: on between the ('mps','on') and ('mps','off') steps."""
    on = False
    for s in plan[:i]:
        if s[0] == "mps":
            on = s[1] == "on"
    return on


# ----------------------------------------------------------------------------- MPS control
def mps_running():
    r = subprocess.run("echo get_server_list | nvidia-cuda-mps-control", shell=True, capture_output=True, text=True)
    return r.returncode == 0 and "Cannot find MPS control daemon" not in (r.stdout + r.stderr)


def mps_set(on):
    if on and not mps_running():
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
        time.sleep(2)
    if not on and mps_running():
        subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True)
        time.sleep(3)
    return mps_running() == on


# ----------------------------------------------------------------------------- orchestrator
def run(args):
    out = os.path.abspath(args.out)
    for d in ("logs", "prof", "cfg"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    plan = jload(os.path.join(out, "plan.json"))
    if plan is None:
        plan = make_plan(args.smoke)
        jdump(plan, os.path.join(out, "plan.json"))
        jdump({"created": time.strftime("%F %T"), "smoke": args.smoke, "nsys": NSYS, "full": FULL, "seed": SEED},
              os.path.join(out, "run_meta.json"))
    state = jload(os.path.join(out, "state.json"), {"done": {}, "failed": {}})
    with open(os.path.join(out, "orchestrator.pid"), "w") as f:
        f.write(str(os.getpid()))
    print("nsys:", NSYS, flush=True)
    t_start = time.time()
    for i, s in enumerate(plan):
        k = sid(s)
        if k in state["done"]:
            continue
        while shutil.disk_usage(out).free < MIN_FREE_GB * 2**30:
            progress(out, plan, state, f"PAUSED: less than {MIN_FREE_GB} GB free (next {k})", t_start)
            time.sleep(60)
        progress(out, plan, state, k, t_start)
        blk = s[1] if s[0] in ("pprep", "prebatch", "build", "clean", "aprep", "ref", "atime") else \
            (s[2] if s[0] == "prof" else None)
        t0 = time.time()
        if blk and (f"pprep|{blk}" in state["failed"] or f"aprep|{blk}" in state["failed"]
                    or f"prebatch|{blk}" in state["failed"]) and s[0] not in ("pprep", "aprep", "prebatch"):
            ok, msg = False, "skipped: preparation of this block failed"
        else:
            want = need_mps(plan, i) if s[0] != "mps" else None
            if want is not None and not mps_set(want):
                ok, msg = False, f"could not set MPS daemon to {'on' if want else 'off'}"
            else:
                ok, msg = step(out, s)
        dt = round(time.time() - t0, 1)
        if ok:
            state["done"][k] = dt
            state["failed"].pop(k, None)
        else:
            state["failed"][k] = msg[-300:]
            write_rows(os.path.join(out, "failures.csv"), ["ts", "step", "seconds", "error"],
                       [{"ts": time.strftime("%F %T"), "step": k, "seconds": dt, "error": msg[-600:]}])
        jdump(state, os.path.join(out, "state.json"))
    progress(out, plan, state, "FINISHED", t_start)


def sub(out, s, cmd, env_extra=None, timeout=7200):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(THREADS)
    env.update(env_extra or {})
    log = os.path.join(out, "logs", f"{s[2] if s[0] == 'prof' else s[1]}.log")
    with open(log, "a") as f:
        f.write(f"\n===== {time.strftime('%F %T')}  {sid(s)}\n$ {' '.join(cmd)}\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
            if r.returncode == 0:
                return True, ""
            msg = f"exit code {r.returncode}"
        except subprocess.TimeoutExpired:
            msg = "timeout"
    try:
        msg += " | " + " ".join(x.strip() for x in open(log).readlines()[-12:])
    except Exception:
        pass
    return False, msg


def step(out, s):
    kind = s[0]
    if kind == "pprep":
        return sub(out, s, [PV_PY, HERE, "_pprep", out, s[1]])
    if kind == "prebatch":
        return sub(out, s, [PV_PY, HERE, "_prebatch", out, s[1]])
    if kind == "build":
        return sub(out, s, [PV_PY, core.HERE, "_build", out, s[1], s[2], s[3], s[4]], timeout=TIMEOUT["build"])
    if kind == "clean":
        d = cdir(out, s[1])
        shutil.rmtree(os.path.join(d, "engines"), ignore_errors=True)
        for x in os.listdir(d) if os.path.isdir(d) else []:
            if x.startswith("_k"):
                shutil.rmtree(os.path.join(d, x), ignore_errors=True)
        od = os.path.join(d, "onnx")
        for f in os.listdir(od) if os.path.isdir(od) else []:
            if not re.fullmatch(r"(fp32|fp16|int8a|int8b)_b1\.onnx", f):
                os.remove(os.path.join(od, f))
        return True, ""
    if kind == "mps":
        ok = mps_set(s[1] == "on")
        return ok, "" if ok else "MPS daemon did not change state"
    if kind == "check":                    # rule 13: MPS 100 % vs no MPS, base640 @640, eager / eager + graph
        cid = "A_base640_s640"
        ok_all, msgs = True, []
        for b in (1, 32):
            env = {"EXTRA_MPS": s[1]} if s[1] == "off" else {"EXTRA_MPS": "100", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100"}
            ok, m = sub(out, ("check", cid), [PV_PY, HERE, "_atime", out, cid, str(b), "torch", "check"], env)
            ok_all &= ok
            msgs.append(m)
        return ok_all, " ".join(msgs)
    if kind == "aprep":
        return sub(out, s, [PV_PY, HERE, "_aprep", out, s[1]])
    if kind == "ref":
        env = {"EXTRA_MPS": "100", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100"}
        return sub(out, s, [PV_PY, HERE, "_atime", out, s[1], "1", "ref", s[2]], env, timeout=600)
    if kind == "atime":
        _, cid, b, fam, ps, mp = s
        env = {"EXTRA_MPS": mp, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": mp}
        return sub(out, s, [PV_PY, HERE, "_atime", out, cid, b, fam, ps], env, timeout=TIMEOUT["atime"])
    if kind == "prof":
        _, stage, cid, b, rt, mp = s
        tag = f"{stage}_{cid}_b{b}_{rt}_m{mp}"
        base = os.path.join(out, "prof", tag)
        env = {"EXTRA_MPS": mp}
        if mp != "off":
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = mp
        if rt.startswith("compile"):
            env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        cmd = [NSYS, "profile", "-t", "cuda,nvtx", "-s", "none", "--cpuctxsw=none", "--cuda-graph-trace=node",
               "-o", base, "--force-overwrite=true", "--export=sqlite",
               PV_PY, HERE, "_prof", out, cid, b, rt, tag]
        ok, msg = sub(out, s, cmd, env, timeout=TIMEOUT["prof"])
        side = jload(base + ".json", {}) or {}
        cfg, _, sz = resolve(cid)
        rec = {"ts": time.strftime("%F %T"), "stage": stage, "cid": cid, "cfg": cfg, "imgsz": sz, "batch": b,
               "runtime": rt, "mps": mp, "leaf_calls": side.get("leaf_calls", "")}
        if not ok and not side:
            return False, msg
        if side.get("status", "ok") != "ok":
            rec.update(status=side["status"])
        elif not os.path.exists(base + ".sqlite"):
            return False, "nsys produced no sqlite | " + msg
        else:
            rows, names = parse_nsys(base + ".sqlite")
            rec.update(summarize(rows))
            if names:
                jdump(dict(names.most_common()), base + ".kernels.json")
        write_rows(os.path.join(out, "profile.csv"), P_KEYS, [rec])
        for ext in (".sqlite", ".nsys-rep", ".qdstrm"):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        return True, ""
    return False, f"unknown step {s}"


# ----------------------------------------------------------------------------- progress
def progress(out, plan, state, current, t_start):
    done = state["done"]
    avg = collections.defaultdict(list)
    for k, dt in done.items():
        avg[stype(k.split("|"))].append(dt)
    avg = {k: sum(v) / len(v) for k, v in avg.items()}
    rem = [s for s in plan if sid(s) not in done]
    eta = sum(avg.get(stype(s), DEFAULT_S.get(stype(s), 60)) for s in rem)
    cut = next((i for i, s in enumerate(plan) if s[0] == "check"), len(plan))
    stages = {"P (profiling)": plan[:cut], "A (Slice A)": plan[cut:]}
    blocks = collections.OrderedDict()
    for s in plan:
        b = s[2] if s[0] == "prof" else (s[1] if s[0] not in ("check", "mps") else "MPS check")
        d = blocks.setdefault(b, [0, 0])
        d[1] += 1
        d[0] += sid(s) in done
    jdump({"total": len(plan), "done": len(done), "failed": len(state["failed"]), "current": current,
           "eta_s": round(eta), "work_s": round(sum(done.values())), "avg_s": {k: round(v, 1) for k, v in avg.items()},
           "stages": {k: [sum(sid(s) in done for s in v), len(v)] for k, v in stages.items()},
           "blocks": blocks, "updated": time.strftime("%F %T")}, os.path.join(out, "progress.json"))


def show(out, once=False):
    while True:
        P = jload(os.path.join(out, "progress.json"))
        if not once:
            os.system("clear")
        if not P:
            print("no progress.json yet in", out)
        else:
            p = P["done"] / max(1, P["total"])
            print(f"Paper 2 profiling + Slice A   {time.strftime('%F %T')}   (updated {P['updated']})\n")
            print(f"Overall  [{bar(p)}] {p * 100:5.1f}%   {P['done']}/{P['total']} steps   failed {P['failed']}")
            for k, (d, t) in P["stages"].items():
                print(f"Stage {k:14s} [{bar(d / max(1, t), 30)}] {d}/{t}")
            print(f"Work time {hms(P['work_s'])}   ETA {hms(P['eta_s'])}")
            print(f"Current  {P['current']}")
            print(f"Disk     {shutil.disk_usage(out).free / 2**30:.1f} GB free    MPS daemon {'on' if mps_running() else 'off'}")
            for b, (d, t) in P["blocks"].items():
                if 0 < d < t:
                    print(f"  {b:16s} [{bar(d / t, 20)}] {d}/{t}")
            n_fin = sum(1 for d, t in P["blocks"].values() if d == t)
            print(f"\n  blocks finished: {n_fin}/{len(P['blocks'])}")
            fl = os.path.join(out, "failures.csv")
            if os.path.exists(fl):
                rows = list(csv.DictReader(open(fl)))
                if rows:
                    print(f"  last failure: {rows[-1]['step']}  {rows[-1]['error'][:160]}")
            pf = os.path.join(out, "profile.csv")
            if os.path.exists(pf):
                rows = list(csv.DictReader(open(pf)))
                bad = [r for r in rows if r.get("status") != "ok"]
                print(f"  profiled cells: {len(rows)}   not ok: {len(bad)}" + (f"  (last: {bad[-1]['cid']} b{bad[-1]['batch']} {bad[-1]['runtime']} {bad[-1]['status']})" if bad else ""))
            if P["current"] == "FINISHED":
                print("\n✓ FINISHED")
                return
        alive = False
        try:
            os.kill(int(open(os.path.join(out, "orchestrator.pid")).read()), 0)
            alive = True
        except Exception:
            pass
        if not alive and not once:
            print("\n⚠ orchestrator is not running; restart with the same 'run' command to resume")
            return
        if once:
            return
        time.sleep(5)


# ----------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        role, a = sys.argv[1], sys.argv[2:]
        {"_pprep": lambda: w_pprep(a[0], a[1]), "_prebatch": lambda: w_prebatch(a[0], a[1]),
         "_aprep": lambda: w_aprep(a[0], a[1]), "_prof": lambda: w_prof(a[0], a[1], int(a[2]), a[3], a[4]),
         "_atime": lambda: w_atime(a[0], a[1], int(a[2]), a[3], a[4])}[role]()
        return
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--smoke", action="store_true")
    for c in ("watch", "status"):
        sp.add_parser(c).add_argument("out")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        try:
            show(os.path.abspath(args.out), once=args.cmd == "status")
        except KeyboardInterrupt:
            print("\n(left the viewer; the job keeps running)")


if __name__ == "__main__":
    main()
