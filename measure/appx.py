#!/usr/bin/env python3
"""
appx.py — Paper 2, last measurements on the RTX 4090 (reuses core.py v7 unchanged).

Part I  (inspection, exploratory): why INT8 separates the D4 pairs. For afterA, B-narrow, A-wide and c1pos at
        batch 8, TensorRT engines (INT8 a, INT8 b, FP16) are built with detailed profiling verbosity; the engine
        inspector gives the precision TensorRT chose for every layer, a layer profiler gives per-layer time, and
        the Q/DQ graphs are scanned for which convolutions ModelOpt quantized.
Part II (appendix of the protocol, latency only, context): base640, c1pos and YOLO11-free reference YOLO26-s
        (random weights, 10 classes) at 640, batch 1 and 8: eager FP32, eager + CUDA Graph, TensorRT FP16
        (graph / no graph), eager BF16, TensorRT BF16 + graph, ONNX Runtime on the CPU.

  python appx.py check                         environment, weights, calibrated INT8 graphs, YOLO26 support
  python appx.py run --out DIR [--smoke]        orchestrator (resumable; run in background)
  python appx.py watch DIR | status DIR         progress
"""
import collections, json, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                                                    # noqa: E402

HERE = os.path.abspath(__file__)
PY = os.environ.get("PV_PY") or sys.executable
FULL = os.environ.get("FULL_DIR", "/workspace/paper2_core/full")    # core run: calibrated batch-1 INT8 graphs
I_CFGS = ["afterA", "Bnarrow", "Awide", "c1pos"]
I_BATCH = 8
A_CFGS = ["base640", "c1pos", "y26s"]
A_BATCHES = (1, 8)
CPU_WARM, CPU_ITERS = 10, 100        # ONNX Runtime CPU: fewer passes than the GPU cells (deviation, see README)
core.BATCHES = A_BATCHES             # core.w_prep exports batch 1 and 8 only
core.CFG["y26s"] = ("y26s", "__yolo26s__", 640)
core.THREADS = min(14, len(os.sched_getaffinity(0)))
core.DEFAULT_S.update({"iprep": 120, "ibuild": 180, "itime": 60, "icleanup": 1, "prep": 30, "build": 170,
                       "bbuild": 200, "time_torch": 60, "time_trt": 60, "time_bf16": 40, "time_ortcpu": 120})

_orig_load = core.load_model


def load_model(run_name):
    """core.load_model for trained configurations; YOLO26-s is built from its YAML with random weights."""
    if run_name != "__yolo26s__":
        return _orig_load(run_name)
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel("yolo26s.yaml", nc=10, verbose=False).float().eval()
    m.fuse(verbose=False)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.cuda()


core.load_model = load_model


# ----------------------------------------------------------------------------- plan
def make_plan(cfgs, nbuild, shuffle_cfgs, smoke=False):
    import random
    rng = random.Random(f"{core.SEED}-appx")
    plan = []
    for cfg in (I_CFGS[:1] if smoke else I_CFGS):
        plan.append(("iprep", cfg))
        for prec in ("int8a", "int8b", "fp16"):
            plan.append(("ibuild", cfg, prec, I_BATCH))
        plan.append(("itime", cfg))
        plan.append(("icleanup", cfg))
    batches = (1,) if smoke else A_BATCHES
    for cfg in (A_CFGS[-1:] if smoke else A_CFGS):     # smoke: YOLO26-s (base640 was checked in smoke 1)
        plan.append(("prep", cfg))
        builds = [(kind, cfg, prec, b, k) for kind, prec in (("build", "fp16"), ("bbuild", "bf16"))
                  for b in batches for k in range(1, nbuild + 1)]
        rng.shuffle(builds)
        plan += builds
        plan.append(("ref", cfg, "start"))
        for ps in (1, 2):
            steps = [("time", cfg, b, fam, ps) for b in batches for fam in ("torch", "trt", "bf16", "ortcpu")]
            rng.shuffle(steps)
            plan += steps
            plan.append(("ref", cfg, "mid" if ps == 1 else "end"))
        plan.append(("cleanup", cfg))
    return plan


def cmd_for(out, s):
    k = s[0]
    if k in ("iprep", "itime", "icleanup"):
        return [PY, HERE, "_" + k, out, s[1]]
    if k == "ibuild":
        return [PY, HERE, "_ibuild", out, s[1], s[2], str(s[3])]
    if k == "prep":
        return [PY, HERE, "_prep", out, s[1]]
    if k == "build":
        return [PY, HERE, "_build", out, s[1], s[2], str(s[3]), str(s[4])]
    if k == "bbuild":
        return [PY, HERE, "_bbuild", out, s[1], str(s[3]), str(s[4])]
    if k == "time":
        return [PY, HERE, "_time", out, s[1], str(s[2]), s[3], str(s[4])]
    if k == "ref":
        return [PY, HERE, "_time", out, s[1], "1", "ref", s[2]]
    return None


def step_type(s):
    k = s[0]
    if k == "time":
        return f"time_{s[3]}"
    return k


core.cmd_for = cmd_for
core.step_type = step_type


# ----------------------------------------------------------------------------- Part I workers
def idir(out, cfg):
    return os.path.join(out, "insp", cfg)


def w_iprep(out, cfg):
    """FP32/FP16 exports at batch 1 and 8 (core.w_prep, checked against the core checksums), the calibrated batch-1
    INT8 graphs of the core run, re-batched to 8 exactly as in the core layer (core.rebatch + check)."""
    import hashlib
    import numpy as np
    import torch  # noqa: F401
    d = idir(out, cfg)
    os.makedirs(os.path.join(d, "onnx"), exist_ok=True)
    core.w_prep(d, cfg)                       # writes into d/cfg/<cfg>/onnx
    od = os.path.join(core.cdir(d, cfg), "onnx")
    src = os.path.join(core.cdir(FULL, cfg), "onnx")
    q = core.jload(os.path.join(core.cdir(FULL, cfg), "quant.json"), {}) or {}
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    chk = {k: md5(os.path.join(od, k + ".onnx")) == v for k, v in (q.get("fp32_md5") or {}).items()
           if os.path.exists(os.path.join(od, k + ".onnx"))}
    for p in ("int8a", "int8b"):
        shutil.copyfile(os.path.join(src, f"{p}_b1.onnx"), os.path.join(od, f"{p}_b1.onnx"))
    s = core.CFG[cfg][2]
    x = np.random.default_rng(core.SEED).random((I_BATCH, 3, s, s), dtype=np.float32)
    rec = {"fp32_md5_match_core": chk}
    for p in ("int8a", "int8b"):
        dst = os.path.join(od, f"{p}_b{I_BATCH}.onnx")
        r = core.rebatch(os.path.join(od, f"{p}_b1.onnx"), os.path.join(od, "fp32_b1.onnx"),
                         os.path.join(od, f"fp32_b{I_BATCH}.onnx"), dst + ".tmp.onnx")
        c = core.check_batch(os.path.join(od, f"{p}_b1.onnx"), dst + ".tmp.onnx", x, 2)
        if c["median"] > 1e-3 or c["p99"] > 5e-2:
            raise RuntimeError(f"{cfg} {p}: rebatched graph does not reproduce batch 1: {c}")
        os.replace(dst + ".tmp.onnx", dst)
        rec[p] = {"constants": r["batch_dependent_constants"], "vs_b1": c}
    rec["onnx_quantized_convs"] = onnx_qscan(od, cfg)
    core.jdump(rec, os.path.join(d, "prep.json"))
    print("IPREP_OK", cfg, chk)


def _module_of(name):
    m = re.search(r"model\.(\d+)", name or "")
    return int(m.group(1)) if m else -1


def onnx_qscan(od, cfg):
    """Which Conv nodes take a DequantizeLinear output as data input, per top-level module index."""
    import onnx
    res = {}
    for p in ("int8a", "int8b"):
        g = onnx.load(os.path.join(od, f"{p}_b1.onnx"), load_external_data=False).graph
        dq_out = {n.output[0] for n in g.node if n.op_type == "DequantizeLinear"}
        per = collections.defaultdict(lambda: [0, 0])
        for n in g.node:
            if n.op_type == "Conv":
                mod = _module_of(n.name)
                per[mod][1] += 1
                per[mod][0] += int(n.input[0] in dq_out)
        res[p] = {str(k): {"quantized": v[0], "conv": v[1]} for k, v in sorted(per.items())}
        res[p]["total"] = {"quantized": sum(v[0] for v in per.values()), "conv": sum(v[1] for v in per.values())}
    return res


def w_ibuild(out, cfg, prec, b):
    trt = core.import_trt()
    d = idir(out, cfg)
    od = os.path.join(core.cdir(d, cfg), "onnx")
    ed = os.path.join(d, "engines")
    os.makedirs(ed, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(os.path.join(od, f"{prec}_b{b}.onnx")):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfgb = builder.create_builder_config()
    cfgb.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    cfgb.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    t0 = time.time()
    blob = builder.build_serialized_network(net, cfgb)
    if blob is None:
        raise RuntimeError("build returned None")
    with open(os.path.join(ed, f"{prec}_b{b}.plan"), "wb") as f:
        f.write(blob)
    eng = trt.Runtime(logger).deserialize_cuda_engine(blob)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    core.jdump(info, os.path.join(d, f"layers_{prec}_b{b}.json"))
    print("IBUILD_OK", cfg, prec, b, round(time.time() - t0, 1), "s", len(info.get("Layers", [])), "layers")


def _dtype(fmt):
    """Precision from TensorRT's 'Format/Datatype' string (wording differs between versions)."""
    f = (fmt or "").lower()
    for keys, v in ((("int8",), "int8"), (("fp8", "e4m3"), "fp8"), (("bf16", "bfloat16"), "bf16"),
                    (("fp16", "half"), "fp16"), (("fp32", "float"), "fp32"), (("int32",), "int32"),
                    (("int64",), "int64"), (("bool",), "bool")):
        if any(k in f for k in keys):
            return v
    return "other"


def _layer_meta(info):
    meta = {}
    for L in info.get("Layers", []):
        if isinstance(L, dict):
            outs = L.get("Outputs") or []
            fmt = ""
            if outs and isinstance(outs[0], dict):
                fmt = outs[0].get("Format/Datatype") or next((str(v) for k, v in outs[0].items()
                                                             if "format" in k.lower() or "type" in k.lower()), "")
            meta[L.get("Name")] = (L.get("LayerType"), _dtype(fmt), L.get("TacticName", ""), str(fmt)[:80])
    return meta


def w_itime(out, cfg):
    """Per-layer time with a TensorRT layer profiler (no CUDA Graph; 20 warm-up and 100 profiled passes), joined
    with the inspector's layer precision; one summary row per engine."""
    import torch
    trt = core.import_trt()
    d = idir(out, cfg)
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    TDT = {trt.float32: torch.float32, trt.float16: torch.float16}

    class Prof(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.t = collections.defaultdict(list)

        def report_layer_time(self, name, ms):
            self.t[name].append(ms)

    rows, summ = [], []
    for prec in ("int8a", "int8b", "fp16"):
        b = I_BATCH
        with open(os.path.join(d, "engines", f"{prec}_b{b}.plan"), "rb") as f:
            eng = rt.deserialize_cuda_engine(f.read())
        ctx = eng.create_execution_context()
        bufs = {}
        for i in range(eng.num_io_tensors):
            t = eng.get_tensor_name(i)
            bufs[t] = torch.randn(tuple(eng.get_tensor_shape(t)), device="cuda").to(TDT[eng.get_tensor_dtype(t)])
            ctx.set_tensor_address(t, bufs[t].data_ptr())
        st = torch.cuda.Stream()
        # plain timing first (no profiler attached)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(50):
            ctx.execute_async_v3(st.cuda_stream)
        st.synchronize()
        ts = []
        for _ in range(300):
            e0.record(st); ctx.execute_async_v3(st.cuda_stream); e1.record(st); st.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        prof = Prof()
        ctx.profiler = prof
        for _ in range(20):
            ctx.execute_async_v3(st.cuda_stream); st.synchronize()
        prof.t.clear()
        for _ in range(100):
            ctx.execute_async_v3(st.cuda_stream); st.synchronize()
        info = core.jload(os.path.join(d, f"layers_{prec}_b{b}.json"), {}) or {}
        meta = _layer_meta(info)
        tot = collections.Counter()
        for name, v in prof.t.items():
            v = sorted(v)
            ms = v[len(v) // 2]
            lt, dt, tac, fmt = meta.get(name, ("?", "?", "", ""))
            mods = sorted({_module_of(x) for x in re.findall(r"model\.\d+", name)})
            rows.append({"cfg": cfg, "prec": prec, "batch": b, "layer": name[:300], "type": lt, "out_dtype": dt,
                         "tactic": (tac or "")[:120], "fmt": fmt, "median_ms": round(ms, 5), "modules": " ".join(map(str, mods))})
            tot[dt] += ms
        n_by = collections.Counter(m[1] for m in meta.values())
        summ.append({"cfg": cfg, "prec": prec, "batch": b, "latency_median_ms": round(ts[len(ts) // 2], 4),
                     "profiled_sum_ms": round(sum(tot.values()), 4), "n_layers": len(meta),
                     **{f"n_{k}": v for k, v in n_by.items()}, **{f"ms_{k}": round(v, 4) for k, v in tot.items()}})
        del ctx, eng
        torch.cuda.empty_cache()
    core.append_rows(out, "insp_layers.csv", rows)
    core.append_rows(out, "insp_summary.csv", summ)
    print("ITIME_OK", cfg, summ)


def w_icleanup(out, cfg):
    shutil.rmtree(os.path.join(idir(out, cfg), "engines"), ignore_errors=True)
    print("ICLEANUP_OK", cfg)


# ----------------------------------------------------------------------------- Part II workers
def bf16_onnx(fp16_path, out_path):
    """The FP16 export with every FP16 tensor retyped to BF16 (initializers, constants, casts, value infos); graph
    input and output stay FP32 through one Cast each, so all engines take the same FP32 input tensor."""
    import numpy as np, onnx
    from onnx import TensorProto, numpy_helper as nh, helper

    def bits(a):
        u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

    def retype(t):
        if t.data_type == TensorProto.FLOAT16:
            a = nh.to_array(t).astype(np.float32)
            n = TensorProto(); n.name = t.name; n.dims.extend(t.dims)
            n.data_type = TensorProto.BFLOAT16; n.raw_data = bits(a).tobytes()
            t.CopyFrom(n)

    M = onnx.load(fp16_path)
    g = M.graph
    for t in g.initializer:
        retype(t)
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                retype(a.t)
            if n.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.BFLOAT16
    for vi in list(g.value_info):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.BFLOAT16
    iname, oname = g.input[0].name, g.output[0].name
    for n in g.node:
        for i, v in enumerate(n.input):
            if v == iname:
                n.input[i] = iname + "_bf16"
        for i, v in enumerate(n.output):
            if v == oname:
                n.output[i] = oname + "_bf16"
    g.node.insert(0, helper.make_node("Cast", [iname], [iname + "_bf16"], to=TensorProto.BFLOAT16, name="in_cast_bf16"))
    g.node.append(helper.make_node("Cast", [oname + "_bf16"], [oname], to=TensorProto.FLOAT, name="out_cast_fp32"))
    g.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    g.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.save(M, out_path)


def w_bbuild(out, cfg, b, k):
    trt = core.import_trt()
    d = core.cdir(out, cfg)
    src = os.path.join(d, "onnx", f"bf16_b{b}.onnx")
    if not os.path.exists(src):
        bf16_onnx(os.path.join(d, "onnx", f"fp16_b{b}.onnx"), src + ".tmp.onnx")
        os.replace(src + ".tmp.onnx", src)
    os.makedirs(os.path.join(d, "engines"), exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(src):
        raise RuntimeError("BF16 graph not parsed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfgb = builder.create_builder_config()
    cfgb.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if k == 1:
        cfgb.profiling_verbosity = trt.ProfilingVerbosity.DETAILED      # layer precisions are recorded for build 1
    t0 = time.time()
    blob = builder.build_serialized_network(net, cfgb)
    if blob is None:
        raise RuntimeError("BF16 build returned None")
    bs = round(time.time() - t0, 1)
    dst = os.path.join(d, "engines", f"bf16_b{b}_k{k}.plan")
    with open(dst + ".tmp", "wb") as f:
        f.write(blob)
    os.replace(dst + ".tmp", dst)
    rec = {"ts": time.strftime("%F %T"), "cfg": cfg, "prec": "bf16", "batch": b, "build": k, "build_s": bs,
           "engine_mb": round(blob.nbytes / 2**20, 1), "trt": trt.__version__,
           "method": "strongly typed, BF16-retyped FP16 export"}
    if k == 1:
        rec.update(bf16_check(trt, logger, blob, d, cfg, b))
    core.append_rows(out, "builds.csv", [rec])
    print("BBUILD_OK", rec)


def bf16_check(trt, logger, blob, d, cfg, b):
    """Layer precisions from the engine inspector and the output against the FP32 export (ONNX Runtime CUDA) on
    one random input: max |diff| / max |y_fp32|."""
    import numpy as np, torch
    eng = trt.Runtime(logger).deserialize_cuda_engine(blob)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    core.jdump(info, os.path.join(d, f"layers_bf16_b{b}.json"))
    n = collections.Counter(m[1] for m in _layer_meta(info).values())
    s = core.CFG[cfg][2]
    x = np.random.default_rng(core.SEED).standard_normal((b, 3, s, s)).astype(np.float32)
    ref = core._run(os.path.join(d, "onnx", f"fp32_b{b}.onnx"), x)
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        t = eng.get_tensor_name(i)
        bufs[t] = torch.zeros(tuple(eng.get_tensor_shape(t)), dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(t, bufs[t].data_ptr())
    bufs["images"].copy_(torch.from_numpy(x))
    st = torch.cuda.Stream()
    ctx.execute_async_v3(st.cuda_stream)
    st.synchronize()
    y = bufs["out"].cpu().numpy()
    rel = float(np.max(np.abs(y - ref)) / max(1e-6, float(np.max(np.abs(ref)))))
    return {"rel_diff_vs_fp32": round(rel, 5), **{f"layers_{k_}": v for k_, v in n.items()}}


def w_time_bf16(out, cfg, b, ps):
    import random, torch
    torch.set_num_threads(core.THREADS)
    m = load_model(core.CFG[cfg][1]).to(torch.bfloat16)
    s = core.CFG[cfg][2]
    x = torch.randn(b, 3, s, s, device="cuda").to(torch.bfloat16)
    reps = core.REPS_P1 if ps == 1 else core.REPS_P2
    rows = []
    with torch.no_grad():
        m(x)
        for rep in range(1, reps + 1):
            rows.append(core.row(cfg, b, "eager_bf16", 1, ps, rep, core.time_fn(lambda: m(x))))
    core.append_rows(out, "latency.csv", rows)
    print("TIME_OK", cfg, b, "bf16", ps, len(rows))


def w_time_ortcpu(out, cfg, b, ps):
    import statistics as st, numpy as np, onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = core.THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(core.cdir(out, cfg), "onnx", f"fp32_b{b}.onnx"), so,
                                providers=["CPUExecutionProvider"])
    s = core.CFG[cfg][2]
    xn = np.random.default_rng(core.SEED).standard_normal((b, 3, s, s)).astype(np.float32)
    reps = core.REPS_P1 if ps == 1 else core.REPS_P2
    rows = []
    for rep in range(1, reps + 1):
        for _ in range(CPU_WARM):
            sess.run(None, {"images": xn})
        ts = []
        for _ in range(CPU_ITERS):
            t0 = time.perf_counter(); sess.run(None, {"images": xn}); ts.append((time.perf_counter() - t0) * 1e3)
        r = core.stats(ts); r["wall_median_ms"] = r["median_ms"]
        rows.append(core.row(cfg, b, "ort_cpu", 1, ps, rep, r, timing=f"wall_clock_{CPU_ITERS}_passes"))
    core.append_rows(out, "latency.csv", rows)
    print("TIME_OK", cfg, b, "ortcpu", ps, len(rows))


# ----------------------------------------------------------------------------- check
def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def check():
    ok = True

    def line(tag, good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'OK' if good else '!!'}] {tag:14s} {msg}")

    print("Appendix / inspection environment check\n")
    print("  GPU          ", sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"))
    print("  CPU          ", sh("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'"), "| threads", core.THREADS)
    other = sh("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader")
    line("GPU idle", not other, other or "no other GPU processes")
    import torch
    line("torch", torch.cuda.is_available(), f"{torch.__version__}, {torch.cuda.get_device_name(0)}, "
         f"bf16 {'supported' if torch.cuda.is_bf16_supported() else 'NOT supported'}")
    trt = core.import_trt()
    line("tensorrt", True, trt.__version__)
    for cfg in I_CFGS:
        src = os.path.join(core.cdir(FULL, cfg), "onnx")
        have = all(os.path.exists(os.path.join(src, f"{p}_b1.onnx")) for p in ("int8a", "int8b"))
        line(f"INT8 {cfg}", have, src if have else f"missing calibrated INT8 graphs in {src}")
    for cfg in sorted(set(I_CFGS + A_CFGS)):
        try:
            m = load_model(core.CFG[cfg][1])
            line(cfg, True, f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params")
            del m
        except Exception as ex:
            line(cfg, False, repr(ex)[:200])
    import onnxruntime as ort
    line("ORT CPU", "CPUExecutionProvider" in ort.get_available_providers(), ort.__version__)
    free = shutil.disk_usage(os.path.dirname(FULL)).free / 2**30
    line("disk", free > 8, f"{free:.1f} GB free")
    print("\nCHECK PASSED" if ok else "\nCHECK FOUND PROBLEMS (lines marked !!) - send this output")


# ----------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        role, a = sys.argv[1], sys.argv[2:]
        if role == "_iprep":
            w_iprep(a[0], a[1])
        elif role == "_ibuild":
            w_ibuild(a[0], a[1], a[2], int(a[3]))
        elif role == "_itime":
            w_itime(a[0], a[1])
        elif role == "_icleanup":
            w_icleanup(a[0], a[1])
        elif role == "_prep":
            core.w_prep(a[0], a[1])
        elif role == "_build":
            core.w_build(a[0], a[1], a[2], int(a[3]), int(a[4]))
        elif role == "_bbuild":
            w_bbuild(a[0], a[1], int(a[2]), int(a[3]))
        elif role == "_time":
            fam = a[3]
            if fam == "bf16":
                w_time_bf16(a[0], a[1], int(a[2]), int(a[4]))
            elif fam == "ortcpu":
                w_time_ortcpu(a[0], a[1], int(a[2]), int(a[4]))
            else:
                core.w_time(a[0], a[1], int(a[2]), fam, a[4])
        return
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check")
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--builds", type=int, default=3)
    r.add_argument("--smoke", action="store_true")
    for n in ("watch", "status"):
        sp.add_parser(n).add_argument("out")
    args = ap.parse_args()
    if args.cmd == "check":
        return check()
    if args.cmd == "run":
        out = os.path.abspath(args.out)
        os.makedirs(out, exist_ok=True)
        builds = 1 if args.smoke else args.builds
        if not os.path.exists(os.path.join(out, "plan.json")):
            plan = make_plan(None, builds, False, smoke=args.smoke)
            core.jdump([list(x) for x in plan], os.path.join(out, "plan.json"))
            cfgs = sorted({s[1] for s in plan})
            core.jdump({"configs": cfgs, "builds": builds, "created": time.strftime("%F %T"), "seed": core.SEED,
                        "share_ab": True, "smoke": args.smoke}, os.path.join(out, "run_meta.json"))
            core.jdump({"layer": "appendix + INT8 inspection", "i_cfgs": I_CFGS, "a_cfgs": A_CFGS, "threads": core.THREADS,
                        "cpu_passes": [CPU_WARM, CPU_ITERS], "full_dir": FULL, "created": time.strftime("%F %T"),
                        "gpu": sh("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader"),
                        "cpu": sh("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'")}, os.path.join(out, "appx_meta.json"))
        cfgs = sorted({s[1] for s in core.jload(os.path.join(out, "plan.json"))})
        core.run(type("A", (), {"out": out, "configs": ",".join(cfgs), "builds": builds, "share_ab": True})())
    elif args.cmd == "watch":
        try:
            core.show(os.path.abspath(args.out))
        except KeyboardInterrupt:
            print("\n(left the viewer; the job keeps running)")
    else:
        core.show(os.path.abspath(args.out), once=True)


if __name__ == "__main__":
    main()
