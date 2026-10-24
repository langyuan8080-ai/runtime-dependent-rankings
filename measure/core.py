#!/usr/bin/env python3
"""
core.py — Paper 2, core layer (pre-registration frozen 2026-10-16).

One file, several roles:
  python core.py run   --out DIR [--configs a,b] [--builds N]   orchestrator (resumable; run in background)
  python core.py watch DIR                                        progress bar + ETA (Ctrl+C leaves the job running)
  python core.py status DIR                                       one-shot summary
Internal workers (started by the orchestrator in fresh processes):
  _prep  _quant  _build  _time

Latency cells: 19 architecture x input combinations x 10 runtime cells x batch {1, 8, 32}.
Runtime cells: eager | eager+CUDA Graph | torch.compile default | torch.compile reduce-overhead |
               ONNX Runtime CUDA (I/O binding; + a with-copies figure) |
               TensorRT FP32 graph | FP16 graph | FP16 no-graph | INT8(a) graph | INT8(b) graph
"""
import argparse, csv, json, os, random, re, shutil, subprocess, sys, time, traceback

RUNS = "/workspace/runs/detect"
DATA = "/workspace/data"
PV_PY = os.environ.get("PV_PY", "/workspace/paper2_venv/bin/python")
QV_PY = os.environ.get("QV_PY", "/workspace/quant_venv/bin/python")
HERE = os.path.abspath(__file__)
BATCHES = (1, 8, 32)
PRECS = ("fp32", "fp16", "int8a", "int8b")
REPS_P1, REPS_P2, REPS_COMPILE = 3, 2, 5
WARM, ITERS = 50, 300
SEED = 20261016
N_CALIB = 512
THREADS = 14
MIN_FREE_GB = 4
MOCK = os.environ.get("MOCK") == "1"

# name, seed-0 run used for weights (latency does not depend on weights; trained weights are needed for INT8 calibration), input
CONFIGS = [
    ("base640", "baseline_scratch", 640), ("p2_640", "stage6_p2fix_s0", 640), ("afterA", "safe_p2_yolo11s", 640),
    ("c1bn", "stage5_c1bn_s0", 640), ("c1pos", "stage3_c1pos", 640), ("Awide", "stage7_Awide_s0", 640),
    ("Bnarrow", "sci_Bnarrow_s0", 640), ("Bplain", "stage7_Bplain_s0", 640), ("Bg4", "sci_Bg4_s0", 640),
    ("seg1", "cap_se_g1_s0", 640), ("seg4", "cap_se_g4_s0", 640), ("safew288", "cap_safew288_s0", 640),
    ("y11m", "yolo11m_s0", 640), ("y11l", "yolo11l_s0", 640),
    ("base960", "hr_base960_s0", 960), ("base1024", "hr_base1024_s0", 1024), ("base1120", "hr_base1120_s0", 1120),
    ("p2_832", "hr_p2_832_s0", 832), ("p2_960", "hr_p2_960_s0", 960),
]
CFG = {c[0]: c for c in CONFIGS}
DEFAULT_S = {"prep": 120, "quant": 2400, "build": 150, "ref": 25, "cleanup": 1,
             "time_torch": 60, "time_ort": 60, "time_trt": 240, "time_compile": 120}
TIMEOUT = {"prep": 3600, "quant": 14400, "build": 3600, "ref": 600, "time": 7200}


# ----------------------------------------------------------------------------- helpers
def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def jload(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def cdir(out, cfg):
    return os.path.join(out, "cfg", cfg)


LAT_KEYS = ["ts", "cfg", "imgsz", "batch", "runtime", "build", "pass", "rep", "median_ms", "mean_ms", "sd_ms",
            "p95_ms", "n", "wall_median_ms", "compile_s", "timing", "block"]


def append_rows(out, name, rows):
    path = os.path.join(out, name)
    new = not os.path.exists(path)
    keys = list(LAT_KEYS) if name == "latency.csv" else []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys if new else _header(path), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _header(path):
    with open(path) as f:
        return next(csv.reader(f))


def step_id(s):
    return "|".join(str(x) for x in s)


def step_type(s):
    k = s[0]
    if k == "time":
        fam = s[3]
        return "time_compile" if fam.startswith("compile") else f"time_{fam}"
    return k


# ----------------------------------------------------------------------------- plan
def make_plan(cfgs, nbuild, shuffle_cfgs):
    rng = random.Random(SEED)
    order = list(cfgs)
    if shuffle_cfgs:
        rng.shuffle(order)
    plan = []
    for cfg in order:
        plan.append(("prep", cfg))
        plan.append(("quant", cfg))
        builds = [("build", cfg, p, b, k) for p in PRECS for b in BATCHES for k in range(1, nbuild + 1)]
        rng.shuffle(builds)
        plan += builds
        plan.append(("ref", cfg, "start"))
        p1 = [("time", cfg, b, fam, 1) for b in BATCHES for fam in ("torch", "ort", "trt")]
        p1 += [("time", cfg, b, f"compile_{m}_k{k}", 1) for b in BATCHES for m in ("default", "ro")
               for k in range(1, nbuild + 1)]
        rng.shuffle(p1)
        plan += p1
        plan.append(("ref", cfg, "mid"))
        p2 = [("time", cfg, b, fam, 2) for b in BATCHES for fam in ("torch", "ort", "trt")]
        rng.shuffle(p2)
        plan += p2
        plan.append(("ref", cfg, "end"))
        plan.append(("cleanup", cfg))
    return plan


# ----------------------------------------------------------------------------- orchestrator
def cmd_for(out, s):
    k = s[0]
    if k == "prep":
        return [PV_PY, HERE, "_prep", out, s[1]]
    if k == "quant":
        return [QV_PY, HERE, "_quant", out, s[1]]
    if k == "build":
        return [PV_PY, HERE, "_build", out, s[1], s[2], str(s[3]), str(s[4])]
    if k == "time":
        return [PV_PY, HERE, "_time", out, s[1], str(s[2]), s[3], str(s[4])]
    if k == "ref":
        return [PV_PY, HERE, "_time", out, s[1], "1", "ref", s[2]]
    return None


def run(args):
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    cfgs = args.configs.split(",") if args.configs else [c[0] for c in CONFIGS]
    for c in cfgs:
        if c not in CFG:
            sys.exit(f"unknown config {c}")
    plan_path = os.path.join(out, "plan.json")
    plan = jload(plan_path)
    if plan is None:
        plan = [list(x) for x in make_plan(cfgs, args.builds, shuffle_cfgs=not args.configs)]
        jdump(plan, plan_path)
        jdump({"configs": cfgs, "builds": args.builds, "created": time.strftime("%F %T"), "seed": SEED,
               "share_ab": bool(args.share_ab)}, os.path.join(out, "run_meta.json"))
    # fixed when the run is created, so a resumed run cannot silently switch procedure
    os.environ["QUANT_SHARE_AB"] = "1" if (jload(os.path.join(out, "run_meta.json"), {}) or {}).get("share_ab") else "0"
    print("QUANT_SHARE_AB =", os.environ["QUANT_SHARE_AB"], flush=True)
    state = jload(os.path.join(out, "state.json"), {"done": {}, "failed": {}})
    with open(os.path.join(out, "orchestrator.pid"), "w") as f:
        f.write(str(os.getpid()))
    t_start = time.time()
    for i, s in enumerate(plan):
        sid = step_id(s)
        if sid in state["done"]:
            continue
        warned = False
        while shutil.disk_usage(out).free < MIN_FREE_GB * 2**30:      # wait instead of failing a build on a full disk
            if not warned:
                write_progress(out, plan, state, current=f"PAUSED: less than {MIN_FREE_GB} GB free on disk "
                               f"(next step {sid}); free space and it continues", t_start=t_start)
                warned = True
            time.sleep(60)
        write_progress(out, plan, state, current=sid, t_start=t_start)
        t0 = time.time()
        if s[0] not in ("prep", "quant") and (f"prep|{s[1]}" in state["failed"] or f"quant|{s[1]}" in state["failed"]):
            # the whole block waits: a resumed run then measures it complete, in one session, after the fix
            ok, msg = False, "skipped: preparation/quantization of this configuration failed"
        else:
            ok, msg = run_step(out, s)
        dt = round(time.time() - t0, 1)
        if ok:
            state["done"][sid] = dt
            state["failed"].pop(sid, None)
        else:
            state["failed"][sid] = msg[-300:]
            append_rows(out, "failures.csv", [{"ts": time.strftime("%F %T"), "step": sid, "seconds": dt, "error": msg[-500:]}])
        jdump(state, os.path.join(out, "state.json"))
    write_progress(out, plan, state, current="FINISHED", t_start=t_start)


def run_step(out, s):
    if s[0] == "cleanup":
        shutil.rmtree(os.path.join(cdir(out, s[1]), "engines"), ignore_errors=True)
        # keep the batch-1 graphs (FP32/FP16 export, calibrated INT8 a/b); batch-8/32 graphs are reproducible from
        # them (re-export / rebatch()) and their checksums are in quant.json
        od = os.path.join(cdir(out, s[1]), "onnx")
        for f in os.listdir(od) if os.path.isdir(od) else []:
            if re.fullmatch(r"(fp32|fp16|int8a|int8b)_b(8|32)\.onnx", f):
                os.remove(os.path.join(od, f))
        return True, ""
    cmd = cmd_for(out, s)
    log = os.path.join(out, "logs", f"{s[1]}.log")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(THREADS)
    if s[0] == "time" and s[3].startswith("compile"):
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"   # each compilation is independent (3 builds per cell)
    if MOCK:
        time.sleep(0.02)
        if s[0] in ("time", "ref"):
            append_rows(out, "latency.csv", [{"ts": time.strftime("%F %T"), "cfg": s[1], "runtime": "mock", "median_ms": 1.0}])
        return True, ""
    with open(log, "a") as f:
        f.write(f"\n===== {time.strftime('%F %T')}  {step_id(s)}\n$ {' '.join(cmd)}\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                               timeout=TIMEOUT.get(s[0], 7200))
            if r.returncode == 0:
                return True, ""
            msg = f"exit code {r.returncode}"
        except subprocess.TimeoutExpired:
            msg = "timeout"
    try:
        with open(log) as f:
            tail = f.readlines()[-12:]
        msg += " | " + " ".join(x.strip() for x in tail)
    except Exception:
        pass
    return False, msg


def write_progress(out, plan, state, current, t_start):
    done = state["done"]
    avg = {}
    for sid, dt in done.items():
        t = step_type(sid.split("|"))
        avg.setdefault(t, []).append(dt)
    avg = {k: sum(v) / len(v) for k, v in avg.items()}
    remaining = [s for s in plan if step_id(s) not in done]
    eta = sum(avg.get(step_type(s), DEFAULT_S.get(step_type(s), 60)) for s in remaining)
    per_cfg = {}
    for s in plan:
        d = per_cfg.setdefault(s[1], [0, 0])
        d[1] += 1
        if step_id(s) in done:
            d[0] += 1
    jdump({"total": len(plan), "done": len(done), "failed": len(state["failed"]), "current": current,
           "eta_s": round(eta), "run_started": t_start, "work_s": round(sum(done.values())),
           "avg_s": {k: round(v, 1) for k, v in avg.items()}, "per_cfg": per_cfg,
           "updated": time.strftime("%F %T")}, os.path.join(out, "progress.json"))


def bar(p, w=40):
    f = int(round(p * w))
    return "#" * f + "-" * (w - f)


def hms(s):
    s = int(max(0, s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def show(out, once=False):
    while True:
        P = jload(os.path.join(out, "progress.json"))
        if not once:
            os.system("clear")
        if not P:
            print("no progress.json yet in", out)
        else:
            p = P["done"] / max(1, P["total"])
            print(f"Paper 2 core layer   {time.strftime('%F %T')}   (updated {P['updated']})\n")
            print(f"Overall  [{bar(p)}] {p * 100:5.1f}%   {P['done']}/{P['total']} steps   failed {P['failed']}")
            print(f"Work time {hms(P['work_s'])}   ETA {hms(P['eta_s'])}  (estimated from measured step times)")
            print(f"Current  {P['current']}")
            print(f"Disk     {shutil.disk_usage(out).free / 2**30:.1f} GB free")
            cur = P["current"].split("|")
            if len(cur) > 1:
                try:
                    with open(os.path.join(out, "logs", f"{cur[1]}.log"), errors="ignore") as f:
                        lines = [l.strip() for l in f.readlines()[-400:] if l.strip()]
                    key = [l for l in lines if re.search(r"QUANT|BUILD_OK|TIME_OK|PREP_OK|REF_OK|Error|error|Traceback", l)]
                    print(f"  last log: {(key or lines or [''])[-1][:150]}")
                except Exception:
                    pass
            print()
            for cfg, (d, t) in P["per_cfg"].items():
                if 0 < d < t or (d == t and t > 0 and once):
                    print(f"  {cfg:9s} [{bar(d / t, 20)}] {d:3d}/{t}")
            n_done_cfg = sum(1 for d, t in P["per_cfg"].values() if d == t)
            print(f"\n  configurations finished: {n_done_cfg}/{len(P['per_cfg'])}")
            if P["avg_s"]:
                print("  mean step time (s): " + ", ".join(f"{k} {v}" for k, v in sorted(P["avg_s"].items())))
            fl = os.path.join(out, "failures.csv")
            if os.path.exists(fl):
                with open(fl) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    print(f"\n  last failure: {rows[-1]['step']}  {rows[-1]['error'][:160]}")
            try:
                hosts = {}
                with open(os.path.join(out, "env.jsonl")) as f:
                    for line in f:
                        e = json.loads(line)
                        hosts.setdefault((e.get("cpu"), e.get("gpu_uuid", "?")), []).append(e["cfg_block"])
                if len(hosts) > 1:
                    print(f"\n  ⚠ {len(hosts)} different machines so far (CPU / GPU UUID):")
                    for (c, g), blocks in hosts.items():
                        print(f"     {c} / {g}: {sorted(set(blocks))}")
            except Exception:
                pass
            if P["current"] == "FINISHED":
                print("\n✓ FINISHED")
                return
        alive = False
        try:
            pid = int(open(os.path.join(out, "orchestrator.pid")).read())
            os.kill(pid, 0)
            alive = True
        except Exception:
            pass
        if not alive and not once:
            print("\n⚠ orchestrator is not running (finished, stopped or crashed); restart with the same 'run' command to resume")
            return
        if once:
            return
        time.sleep(5)


# ----------------------------------------------------------------------------- shared worker code
def load_model(run_name):
    import torch, torch.nn as nn
    from ultralytics import YOLO
    m = YOLO(os.path.join(RUNS, run_name, "weights", "best.pt")).model.float().eval()
    for mod in m.modules():                      # attributes added after old SAFE checkpoints were saved
        if type(mod).__name__ == "SAFE":
            if not hasattr(mod, "sa"):
                mod.sa = nn.Identity()
            for a, v in (("use_spatial", False), ("gamma_per_channel", False), ("use_multiscale", True),
                         ("use_attention", True), ("use_gamma", True)):
                if not hasattr(mod, a):
                    setattr(mod, a, v)
    m.fuse(verbose=False)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.cuda()


def wrap(m):
    import torch.nn as nn

    class Wrap(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, x):
            y = s.m(x)
            return y[0] if isinstance(y, (tuple, list)) else y
    return Wrap(m).eval()


def stats(ts):
    import statistics as st
    ts = sorted(ts)
    return {"median_ms": round(st.median(ts), 4), "mean_ms": round(st.mean(ts), 4), "sd_ms": round(st.stdev(ts), 4),
            "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 4), "n": len(ts)}


def time_fn(fn, stream=None, warm=WARM, iters=ITERS):
    """CUDA events around each pass + synchronization after each pass; host wall clock recorded alongside."""
    import torch
    s = stream or torch.cuda.current_stream()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ev, wall = [], []
        for _ in range(iters):
            t0 = time.perf_counter()
            e0.record(s)
            fn()
            e1.record(s)
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1e3)
            ev.append(e0.elapsed_time(e1))
    r = stats(ev)
    r["wall_median_ms"] = round(sorted(wall)[len(wall) // 2], 4)
    return r


def time_wall(fn, warm=WARM, iters=ITERS):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    r = stats(ts)
    r["wall_median_ms"] = r["median_ms"]
    return r


def row(cfg, b, runtime, build, ps, rep, r, **extra):
    d = {"ts": time.strftime("%F %T"), "cfg": cfg, "imgsz": CFG[cfg][2], "batch": b, "runtime": runtime,
         "build": build, "pass": ps, "rep": rep}
    d.update(r)
    d.update(extra)
    return d


# ----------------------------------------------------------------------------- _prep
def w_prep(out, cfg):
    import copy, glob, torch
    name, run_name, s = CFG[cfg]
    d = cdir(out, cfg)
    os.makedirs(os.path.join(d, "onnx"), exist_ok=True)
    calib_path = os.path.join(out, "calib_list.json")
    if not os.path.exists(calib_path):
        files = sorted(glob.glob(os.path.join(DATA, "VisDrone2019-DET-train", "images", "*.jpg")))
        random.Random(SEED).shuffle(files)
        jdump(files[:N_CALIB], calib_path)
    m = load_model(run_name)
    meta = {"cfg": cfg, "run": run_name, "imgsz": s, "detect_idx": len(m.model) - 1,
            "params": sum(p.numel() for p in m.parameters()), "torch": torch.__version__}
    w = wrap(m)
    m16 = copy.deepcopy(m).half()
    w16 = wrap(m16)
    for b in BATCHES:
        x = torch.zeros(b, 3, s, s, device="cuda")
        with torch.no_grad():
            w(x)                                    # Detect caches anchors for this shape
            w16(x.half())
        for tag, net, inp in (("fp32", w, x), ("fp16", w16, x.half())):
            p = os.path.join(d, "onnx", f"{tag}_b{b}.onnx")
            if not os.path.exists(p):
                torch.onnx.export(net, (inp,), p + ".tmp", opset_version=17, input_names=["images"],
                                  output_names=["out"], dynamo=False)
                os.replace(p + ".tmp", p)
        torch.cuda.empty_cache()
    jdump(meta, os.path.join(d, "meta.json"))
    print("PREP_OK", meta)


# ----------------------------------------------------------------------------- Q/DQ scale transplant helpers (quant_venv)
QDQ = ("QuantizeLinear", "DequantizeLinear")


def _consts(g):
    m = {t.name: ("init", t) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    m[n.output[0]] = ("const", a)
    return m


def _arr(cm, name):
    from onnx import numpy_helper as nh
    kind, obj = cm[name]
    return nh.to_array(obj if kind == "init" else obj.t)


def _set(cm, name, arr):
    from onnx import numpy_helper as nh
    kind, obj = cm[name]
    t = nh.from_array(arr, name)
    if kind == "init":
        obj.CopyFrom(t)
    else:
        t.name = obj.t.name
        obj.t.CopyFrom(t)


def _qnodes(g):
    m = {n.output[0]: n for n in g.node if n.op_type in QDQ}       # keyed by output tensor (unique in SSA)
    return m


def transplant(src_path, dst_path, out_path, require_equal_sets=True):
    """Copy every Q/DQ scale and zero point from src (calibrated) into dst (same graph, other batch size or a
    subset of the Q/DQ nodes). Fails loudly if the Q/DQ structure does not correspond."""
    import numpy as np, onnx
    S, T = onnx.load(src_path), onnx.load(dst_path)
    sq, tq = _qnodes(S.graph), _qnodes(T.graph)
    missing = sorted(set(tq) - set(sq))
    if missing:
        raise RuntimeError(f"{len(missing)} Q/DQ nodes of {os.path.basename(dst_path)} absent in source, e.g. {missing[:3]}")
    if require_equal_sets and set(tq) != set(sq):
        raise RuntimeError(f"Q/DQ node sets differ: source {len(sq)}, target {len(tq)}")
    sc, tc = _consts(S.graph), _consts(T.graph)
    done, n_set, n_changed, w_cmp, w_diff = set(), 0, 0, 0, 0
    for name, tn in tq.items():
        sn = sq[name]
        if tn.op_type != sn.op_type or tn.input[0] != sn.input[0]:
            raise RuntimeError(f"Q/DQ node {name} differs in type or input between graphs")
        for j in (1, 2):
            if j >= len(tn.input) or not tn.input[j]:
                continue
            a, b = _arr(sc, sn.input[j]), _arr(tc, tn.input[j])
            if a.shape != b.shape or a.dtype != b.dtype:
                raise RuntimeError(f"{name} input {j}: shape/dtype {b.shape}/{b.dtype} vs source {a.shape}/{a.dtype}")
            if tn.input[j] not in done:
                n_changed += int(not np.array_equal(a, b))
                _set(tc, tn.input[j], a)
                done.add(tn.input[j])
                n_set += 1
        if tn.op_type == "DequantizeLinear" and tn.input[0] in tc and sn.input[0] in sc:     # quantized weights
            w_cmp += 1
            w_diff += int(not np.array_equal(_arr(sc, sn.input[0]), _arr(tc, tn.input[0])))
    if w_diff:
        raise RuntimeError(f"{w_diff}/{w_cmp} quantized weight tensors differ between graphs")
    onnx.save(T, out_path)
    return {"qdq_nodes": len(tq), "params_copied": n_set, "params_changed": n_changed, "weights_checked": w_cmp}


def rebatch(q1_path, f1_path, fb_path, out_path):
    """Batch-b INT8 graph = the calibrated batch-1 INT8 graph with its batch-dependent constants and I/O shapes
    taken from the batch-b FP32 export. Batch-dependent constants are found by diffing the two FP32 exports, so
    nothing is guessed; any constant that cannot be matched in the quantized graph is an error."""
    import numpy as np, onnx
    Q, F1, Fb = onnx.load(q1_path), onnx.load(f1_path), onnx.load(fb_path)
    c1, cb, cq = _consts(F1.graph), _consts(Fb.graph), _consts(Q.graph)
    if set(c1) != set(cb):
        raise RuntimeError("FP32 exports at batch 1 and batch b have different constant sets")
    changed = [k for k in c1 if not np.array_equal(_arr(c1, k), _arr(cb, k))]
    missing = [k for k in changed if k not in cq]
    if missing:
        raise RuntimeError(f"{len(missing)} batch-dependent constants not found in the quantized graph: {missing[:5]}")
    for k in changed:
        a, q = _arr(cb, k), _arr(cq, k)
        if a.shape != q.shape:
            raise RuntimeError(f"constant {k}: shape {q.shape} in quantized graph vs {a.shape} in FP32 export")
        _set(cq, k, a.astype(q.dtype))
    shapes = {v.name: v for v in list(Fb.graph.input) + list(Fb.graph.output)}
    for v in list(Q.graph.input) + list(Q.graph.output):
        if v.name not in shapes:
            raise RuntimeError(f"graph I/O {v.name} not in FP32 export")
        src = shapes[v.name].type.tensor_type.shape.dim
        dst = v.type.tensor_type.shape.dim
        if len(src) != len(dst):
            raise RuntimeError(f"rank of {v.name} differs")
        for d_, s_ in zip(dst, src):
            d_.dim_value = s_.dim_value
    del Q.graph.value_info[:]                     # stale batch-1 shapes; TensorRT/ORT re-infer from the inputs
    onnx.save(Q, out_path)
    return {"batch_dependent_constants": len(changed), "names": changed[:20]}


def compare_scales(pa, pb):
    """Relative difference of activation scales on Q nodes shared by two quantized graphs."""
    import numpy as np, onnx
    A, B = onnx.load(pa), onnx.load(pb)
    ca, cb = _consts(A.graph), _consts(B.graph)
    qa = {n.output[0]: n for n in A.graph.node if n.op_type == "QuantizeLinear" and n.input[0] not in ca}
    qb = {n.output[0]: n for n in B.graph.node if n.op_type == "QuantizeLinear" and n.input[0] not in cb}
    common = sorted(set(qa) & set(qb))
    rel = []
    for k in common:
        x, y = _arr(ca, qa[k].input[1]).astype(np.float64), _arr(cb, qb[k].input[1]).astype(np.float64)
        rel.append(float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-12))))
    rel = np.array(rel) if rel else np.zeros(1)
    return {"act_q_a": len(qa), "act_q_b": len(qb), "common": len(common), "identical": int((rel == 0).sum()),
            "median_rel": float(np.median(rel)), "p90_rel": float(np.percentile(rel, 90)), "max_rel": float(rel.max())}


def qcheck(out, cfgs):
    """Run in quant_venv on existing pilot outputs (no GPU needed): how much do activation scales change
    (i) between scheme a and b at the same batch and (ii) between batch sizes calibrated separately?"""
    res = {}
    for cfg in cfgs:
        od = os.path.join(cdir(out, cfg), "onnx")
        have = lambda f: os.path.exists(os.path.join(od, f + ".onnx"))
        pairs = [("int8a_b1", "int8b_b1"), ("int8a_b8", "int8b_b8"), ("int8a_b1", "int8a_b8"),
                 ("int8a_b1", "int8a_b32"), ("int8b_b1", "int8b_b8"), ("int8b_b1", "int8b_b32")]
        for a, b in pairs:
            if have(a) and have(b):
                r = compare_scales(os.path.join(od, a + ".onnx"), os.path.join(od, b + ".onnx"))
                res[f"{cfg}: {a} vs {b}"] = r
                print(f"{cfg:9s} {a:9s} vs {b:9s}  common {r['common']:4d}  identical {r['identical']:4d}  "
                      f"median {r['median_rel']:.2e}  p90 {r['p90_rel']:.2e}  max {r['max_rel']:.2e}", flush=True)
    jdump(res, os.path.join(out, "qcheck.json"))
    print("saved", os.path.join(out, "qcheck.json"))


# ----------------------------------------------------------------------------- _quant (runs in quant_venv)
def w_quant(out, cfg):
    """Deviation-log entry 7 (2026-10-16). Entropy calibration on the same 512 images runs once per scheme, on the
    batch-1 graph. The batch-8/32 INT8 graphs are that same calibrated graph with its batch-dependent constants and
    I/O shapes set for batch b (rebatch(); verified image by image against batch 1). Reasons: calibrating the
    batch-32 graph ran out of GPU memory (c1pos), as did a second ModelOpt pass at batch 32; and separate per-batch
    calibration gave materially different activation scales (pilot qcheck: median relative difference 17-62 %),
    i.e. three different INT8 models per scheme. Now every batch size runs one quantized model per scheme.
    QUANT_SHARE_AB=1 derives scheme b's scales from scheme a (b's Q/DQ nodes are a subset of a's; pilot: shared
    scales identical or within 0.16 %)."""
    import inspect
    import numpy as np, cv2, onnx
    import modelopt
    from modelopt.onnx.quantization import quantize
    d = cdir(out, cfg)
    od = os.path.join(d, "onnx")
    meta = jload(os.path.join(d, "meta.json"))
    s, det = meta["imgsz"], meta["detect_idx"]
    share_ab = os.environ.get("QUANT_SHARE_AB") == "1"

    def letterbox(path):
        im = cv2.imread(path)
        h, w = im.shape[:2]
        r = min(s / h, s / w)
        nw, nh = round(w * r), round(h * r)
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (s - nw) / 2, (s - nh) / 2
        im = cv2.copyMakeBorder(im, round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1),
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

    import onnxruntime as ort
    try:
        ort.preload_dlls()          # loads the CUDA 12 / cuDNN libraries installed as pip packages in quant_venv
    except Exception as ex:
        print("preload_dlls:", repr(ex), flush=True)
    probe = ort.InferenceSession(os.path.join(od, "fp32_b1.onnx"), providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in probe.get_providers() and os.environ.get("ALLOW_CPU_CALIB") != "1":
        raise RuntimeError("ONNX Runtime in quant_venv cannot use CUDA; calibration would silently run on CPU")
    del probe
    n_cal = int(os.environ.get("QUANT_NCALIB", N_CALIB))      # < 512 only for smoke tests outside the measured run
    calib = np.ascontiguousarray(np.stack([letterbox(f) for f in jload(os.path.join(out, "calib_list.json"))[:n_cal]]))
    sig = inspect.signature(quantize).parameters
    info = jload(os.path.join(d, "quant.json"), {}) or {}
    info["modelopt"] = modelopt.__version__
    info["procedure"] = "calibrate batch 1 (entropy, 512 images); batch 8/32 = batch-1 INT8 graph re-batched" + \
                        ("; scheme b derived from scheme a" if share_ab else "")
    pat = rf".*model\.{det}[/.].*"
    excl = {"int8a": [], "int8b": [pat]}

    def clean():                                    # stray files from earlier versions / interrupted runs
        for f in os.listdir(od):
            if not re.fullmatch(r"(fp32|fp16|int8a|int8b)_b\d+\.onnx", f):
                os.remove(os.path.join(od, f))
        shutil.rmtree(os.path.join(d, "qwork"), ignore_errors=True)

    def run_q(scheme, b, method, data, dst):
        # ModelOpt rewrites its input file in place and leaves intermediates next to it, so it only ever sees a
        # private copy in qwork/; the FP32 ONNX used for ORT timing and TRT builds stays exactly as exported.
        wd = os.path.join(d, "qwork")
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd)
        src = os.path.join(wd, f"fp32_b{b}.onnx")
        shutil.copyfile(os.path.join(od, f"fp32_b{b}.onnx"), src)
        kw = dict(onnx_path=src, quantize_mode="int8", calibration_data=data,
                  calibration_method=method, nodes_to_exclude=excl[scheme], high_precision_dtype="fp16",
                  output_path=os.path.join(wd, "out.onnx"), calibration_eps=["cuda:0", "cpu"])
        kw = {k: v for k, v in kw.items() if k in sig}
        try:
            quantize(**kw)
            if not os.path.exists(os.path.join(wd, "out.onnx")):
                raise RuntimeError(f"ModelOpt produced no output; files in qwork: {os.listdir(wd)}")
            shutil.move(os.path.join(wd, "out.onnx"), dst)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        return kw

    def heads(path):
        g = onnx.load(path, load_external_data=False).graph
        return (sum(1 for n in g.node if n.op_type == "QuantizeLinear"),
                sum(1 for n in g.node if n.op_type == "QuantizeLinear" and re.match(pat, n.name)))

    def record(key, rec, path, scheme):
        q_all, q_head = heads(path)
        rec.update({"q_nodes": q_all, "q_nodes_in_head": q_head})
        if scheme == "int8b" and q_head:
            raise RuntimeError(f"scheme b still quantizes {q_head} Detect-head nodes")
        info[key] = rec
        jdump(info, os.path.join(d, "quant.json"))
        print("QUANT", key, rec, flush=True)

    clean()
    import hashlib
    md5 = lambda f: hashlib.md5(open(os.path.join(od, f), "rb").read()).hexdigest()
    before = {f"fp32_b{b}": md5(f"fp32_b{b}.onnx") for b in BATCHES}
    # 1) batch 1: entropy calibration on 512 images (scheme b from scheme a when QUANT_SHARE_AB=1)
    for scheme in ("int8a", "int8b"):
        dst = os.path.join(od, f"{scheme}_b1.onnx")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        if scheme == "int8b" and share_ab:
            print("QUANT start int8b b1 (structure + scales from int8a b1)", flush=True)
            run_q(scheme, 1, "max", calib[:1], dst + ".struct.onnx")
            tr = transplant(os.path.join(od, "int8a_b1.onnx"), dst + ".struct.onnx", dst + ".tmp.onnx",
                            require_equal_sets=False)
            os.remove(dst + ".struct.onnx")
            os.replace(dst + ".tmp.onnx", dst)
            record("int8b_b1", {"seconds": round(time.time() - t0, 1), "calibration": "scales from int8a_b1",
                                "transplant": tr}, dst, scheme)
        else:
            print(f"QUANT start {scheme} b1 (entropy, {len(calib)} images)", flush=True)
            kw = run_q(scheme, 1, "entropy", calib, dst + ".tmp.onnx")
            os.replace(dst + ".tmp.onnx", dst)
            record(f"{scheme}_b1", {"seconds": round(time.time() - t0, 1), "calibration": f"entropy, {len(calib)} images",
                                    "calibration_eps": kw.get("calibration_eps"),
                                    "calibrate_per_node": kw.get("calibrate_per_node", False)}, dst, scheme)
    # 2) batch 8/32: the calibrated batch-1 graph re-batched (no second ModelOpt pass, no calibration at b>1)
    for b in BATCHES[1:]:
        for scheme in ("int8a", "int8b"):
            dst = os.path.join(od, f"{scheme}_b{b}.onnx")
            if os.path.exists(dst):
                continue
            t0 = time.time()
            print(f"QUANT start {scheme} b{b} (batch-1 INT8 graph re-batched)", flush=True)
            tr = rebatch(os.path.join(od, f"{scheme}_b1.onnx"), os.path.join(od, "fp32_b1.onnx"),
                         os.path.join(od, f"fp32_b{b}.onnx"), dst + ".tmp.onnx")
            # numerical check: batch-b graph on the first images must reproduce the batch-1 graph image by image
            c = check_batch(os.path.join(od, f"{scheme}_b1.onnx"), dst + ".tmp.onnx", calib[:b], n=min(b, 4))
            # reference: the same comparison for the FP32 exports (pure floating-point batch effect)
            c["fp32_reference"] = check_batch(os.path.join(od, "fp32_b1.onnx"), os.path.join(od, f"fp32_b{b}.onnx"),
                                              calib[:b], n=min(b, 4))
            tr["vs_b1"] = c
            if c["median"] > 1e-3 or c["p99"] > 5e-2:
                raise RuntimeError(f"{scheme} b{b}: batch-b graph does not reproduce batch 1: {c}")
            os.replace(dst + ".tmp.onnx", dst)
            record(f"{scheme}_b{b}", {"seconds": round(time.time() - t0, 1), "calibration": "batch-1 graph re-batched",
                                      "rebatch": tr}, dst, scheme)
    for b in BATCHES:                               # final consistency: same Q count per scheme for every batch
        for scheme in ("int8a", "int8b"):
            if info[f"{scheme}_b{b}"]["q_nodes"] != info[f"{scheme}_b1"]["q_nodes"]:
                raise RuntimeError(f"{scheme}: Q-node count differs between batch 1 and batch {b}")
    after = {f"fp32_b{b}": md5(f"fp32_b{b}.onnx") for b in BATCHES}
    if after != before:
        raise RuntimeError("FP32 ONNX exports changed during quantization")
    info["fp32_md5"] = after
    jdump(info, os.path.join(d, "quant.json"))
    print("QUANT_OK")


def _run(path, x):
    import numpy as np, onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    dt = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
    return sess.run(None, {sess.get_inputs()[0].name: x.astype(dt)})[0].astype(np.float32)


def check_batch(p1, pb, x, n):
    """Batch-b graph (first n images) vs the batch-1 graph run image by image. INT8 graphs are not bit-stable across
    batch sizes (other conv algorithms -> different float sums -> occasional rounding flips at a Q node), so the
    test separates such isolated flips from a structural error (wrong reshape / mixed images), which would shift
    most outputs: errors are |diff| / max|y_b1| per element."""
    import numpy as np
    yb = _run(pb, x)
    errs = []
    for i in range(n):
        y1 = _run(p1, x[i:i + 1])
        errs.append((np.abs(yb[i:i + 1] - y1) / max(1e-6, float(np.max(np.abs(y1))))).ravel())
    e = np.concatenate(errs)
    return {"median": float(np.median(e)), "p99": float(np.percentile(e, 99)), "max": float(e.max()),
            "frac_gt_1e-3": float((e > 1e-3).mean())}


def same_outputs(pa, pb, x):
    """Two graphs for the same batch size on the same input: max |diff| / max|y_a|."""
    import numpy as np
    ya, yb = _run(pa, x), _run(pb, x)
    return float(np.max(np.abs(ya - yb)) / max(1e-6, float(np.max(np.abs(ya)))))


# ----------------------------------------------------------------------------- _build
def import_trt():
    try:
        import tensorrt as trt
    except ImportError:
        import tensorrt_libs  # noqa: F401
        import tensorrt_bindings as trt
    return trt


def w_build(out, cfg, prec, b, k):
    trt = import_trt()
    d = cdir(out, cfg)
    os.makedirs(os.path.join(d, "engines"), exist_ok=True)
    src = os.path.join(d, "onnx", f"{prec}_b{b}.onnx")
    dst = os.path.join(d, "engines", f"{prec}_b{b}_k{k}.plan")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED") else 0
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(src):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfgb = builder.create_builder_config()
    cfgb.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t0 = time.time()
    blob = builder.build_serialized_network(net, cfgb)
    if blob is None:
        raise RuntimeError("build_serialized_network returned None")
    bs = round(time.time() - t0, 1)
    with open(dst + ".tmp", "wb") as f:
        f.write(blob)
    os.replace(dst + ".tmp", dst)
    append_rows(out, "builds.csv", [{"ts": time.strftime("%F %T"), "cfg": cfg, "prec": prec, "batch": b, "build": k,
                                     "build_s": bs, "engine_mb": round(blob.nbytes / 2**20, 1), "trt": trt.__version__}])
    print("BUILD_OK", cfg, prec, b, k, bs)


# ----------------------------------------------------------------------------- _time
def env_record(out, cfg, tag):
    import platform, torch
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    rec = {"ts": time.strftime("%F %T"), "cfg_block": cfg, "tag": tag, "gpu": torch.cuda.get_device_name(0),
           "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
           # physical identity: a restarted cloud instance can land on another host / another card
           "gpu_uuid": sh("nvidia-smi --query-gpu=uuid --format=csv,noheader"),
           "gpu_pci": sh("nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader"),
           "gpu_vbios": sh("nvidia-smi --query-gpu=vbios_version --format=csv,noheader"),
           "gpu_max_clocks": sh("nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader"),
           "hostname": platform.node(), "boot_id": sh("cat /proc/sys/kernel/random/boot_id"),
           "cpu": sh("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'"),
           "usable_cores": len(os.sched_getaffinity(0)), "torch": torch.__version__, "python": platform.python_version(),
           "mps": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "off"),
           "other_gpu_procs": sh("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader")}
    try:
        rec["tensorrt"] = import_trt().__version__
    except Exception:
        rec["tensorrt"] = "?"
    try:
        import onnxruntime
        rec["onnxruntime"] = onnxruntime.__version__
    except Exception:
        rec["onnxruntime"] = "?"
    with open(os.path.join(out, "env.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def w_time(out, cfg, b, fam, ps):
    import torch
    torch.set_num_threads(THREADS)
    torch.backends.cudnn.benchmark = False
    rng = random.Random(f"{SEED}-{cfg}-{b}-{fam}-{ps}")
    name, run_name, s = CFG[cfg]
    rows = []
    x = torch.randn(b, 3, s, s, device="cuda")

    if fam == "ref":                                   # drift reference: base640, eager, batch 1
        env_record(out, cfg, ps)
        m = load_model(CFG["base640"][1])
        x1 = torch.randn(1, 3, 640, 640, device="cuda")
        with torch.no_grad():
            m(x1)
            r = time_fn(lambda: m(x1))
        append_rows(out, "latency.csv", [dict(row("base640", 1, "ref_eager", 1, f"ref_{ps}", 1, r), block=cfg)])
        print("REF_OK", r)
        return

    ps = int(ps)
    if fam == "torch":
        m = load_model(run_name)
        with torch.no_grad():
            m(x)
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
        units = {"eager": lambda: m(x), "eager_graph": lambda: (sx.copy_(x), g.replay())}
        reps = REPS_P1 if ps == 1 else REPS_P2
        with torch.no_grad():
            for rep in range(1, reps + 1):
                for u in rng.sample(list(units), len(units)):
                    rows.append(row(cfg, b, u, 1, ps, rep, time_fn(units[u])))
        append_rows(out, "latency.csv", rows)

    elif fam.startswith("compile_"):
        mode = "default" if "_default_" in fam else "reduce-overhead"
        k = int(fam.rsplit("_k", 1)[1])
        m = load_model(run_name)
        with torch.no_grad():
            m(x)                                    # anchors cached before compiling -> one graph
            torch._dynamo.reset()
            t0 = time.time()
            cm = torch.compile(m, mode=mode, dynamic=False)
            cm(x)
            torch.cuda.synchronize()
            comp_s = round(time.time() - t0, 1)
            for rep in range(1, REPS_COMPILE + 1):
                rows.append(row(cfg, b, f"compile_{'default' if mode == 'default' else 'ro'}", k, ps, rep,
                                time_fn(lambda: cm(x)), compile_s=comp_s))
        append_rows(out, "latency.csv", rows)

    elif fam == "ort":
        import numpy as np, onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(os.path.join(cdir(out, cfg), "onnx", f"fp32_b{b}.onnx"), so,
                                    providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError("ORT fell back to CPU")
        oshape = [int(v) for v in sess.get_outputs()[0].shape]
        xin = torch.empty_like(x)
        yout = torch.empty(oshape, device="cuda")
        io = sess.io_binding()
        io.bind_input("images", "cuda", 0, np.float32, list(x.shape), xin.data_ptr())
        io.bind_output("out", "cuda", 0, np.float32, oshape, yout.data_ptr())
        xn = x.cpu().numpy()
        # ORT runs on its own CUDA stream: synchronize after the input copy so it cannot race the inference
        units = {"ort_iobinding": lambda: (xin.copy_(x), torch.cuda.synchronize(), sess.run_with_iobinding(io)),
                 "ort_with_copies": lambda: sess.run(None, {"images": xn})}
        reps = REPS_P1 if ps == 1 else REPS_P2
        for rep in range(1, reps + 1):
            for u in rng.sample(list(units), len(units)):
                rows.append(row(cfg, b, u, 1, ps, rep, time_wall(units[u]), timing="wall_clock"))
        append_rows(out, "latency.csv", rows)

    elif fam == "trt":
        trt = import_trt()
        logger = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(logger)
        TDT = {trt.float32: torch.float32, trt.float16: torch.float16}
        edir = os.path.join(cdir(out, cfg), "engines")
        engines = sorted(f for f in os.listdir(edir) if f.endswith(".plan") and f"_b{b}_k" in f)
        if not engines:
            raise RuntimeError(f"no engines for batch {b} in {edir}")
        reps = REPS_P1 if ps == 1 else REPS_P2
        for fn_ in rng.sample(engines, len(engines)):
            prec, _, kk = fn_[:-5].split("_")
            k = int(kk[1:])
            with open(os.path.join(edir, fn_), "rb") as f:
                eng = rt.deserialize_cuda_engine(f.read())
            ctx = eng.create_execution_context()
            bufs = {}
            for i in range(eng.num_io_tensors):
                t = eng.get_tensor_name(i)
                bufs[t] = torch.zeros(tuple(eng.get_tensor_shape(t)), dtype=TDT[eng.get_tensor_dtype(t)], device="cuda")
                ctx.set_tensor_address(t, bufs[t].data_ptr())
            stream = torch.cuda.Stream()
            xsrc = x.to(bufs["images"].dtype)
            inb = bufs["images"]
            with torch.cuda.stream(stream):
                inb.copy_(xsrc)
                ctx.execute_async_v3(stream.cuda_stream)
                stream.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    ctx.execute_async_v3(stream.cuda_stream)
            units = {f"trt_{prec}_graph": lambda: (inb.copy_(xsrc), g.replay())}
            if prec == "fp16":
                units["trt_fp16_nograph"] = lambda: (inb.copy_(xsrc), ctx.execute_async_v3(stream.cuda_stream))
            for rep in range(1, reps + 1):
                for u in rng.sample(list(units), len(units)):
                    rows.append(row(cfg, b, u, k, ps, rep, time_fn(units[u], stream=stream)))
            del g, ctx, eng, bufs
            torch.cuda.empty_cache()
        append_rows(out, "latency.csv", rows)
    else:
        raise ValueError(fam)
    print("TIME_OK", cfg, b, fam, ps, len(rows), "rows")


# ----------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        role, a = sys.argv[1], sys.argv[2:]
        if role == "_prep":
            w_prep(a[0], a[1])
        elif role == "_quant":
            w_quant(a[0], a[1])
        elif role == "_build":
            w_build(a[0], a[1], a[2], int(a[3]), int(a[4]))
        elif role == "_time":
            w_time(a[0], a[1], int(a[2]), a[3], a[4])
        return
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--configs", default="")
    r.add_argument("--builds", type=int, default=3)
    r.add_argument("--share-ab", action="store_true", help="derive INT8 scheme b scales from scheme a")
    w = sp.add_parser("watch")
    w.add_argument("out")
    st = sp.add_parser("status")
    st.add_argument("out")
    qc = sp.add_parser("qcheck")
    qc.add_argument("out")
    qc.add_argument("configs")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "qcheck":
        qcheck(os.path.abspath(args.out), args.configs.split(","))
    elif args.cmd == "watch":
        try:
            show(os.path.abspath(args.out))
        except KeyboardInterrupt:
            print("\n(left the viewer; the job keeps running)")
    else:
        show(os.path.abspath(args.out), once=True)


if __name__ == "__main__":
    main()
