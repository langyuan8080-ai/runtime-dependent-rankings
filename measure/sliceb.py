#!/usr/bin/env python3
"""
sliceb.py — Paper 2, Slice B: the same latency protocol on other GPU generations (pre-registration, matrix row
"Slice B": baseline, +P2, c1pos, B-plain, YOLO11m, D1 pairs; eager / eager + CUDA Graph / TensorRT FP16;
batch 1, 8; native input; GPUs taken in the order RTX 5090, RTX 3060, RTX 3080 Ti -> 3060 not rentable, so
RTX 5090 + RTX 3080 Ti).

Reuses core.py (v7, unchanged) for every measurement: model loading, ONNX export settings, TensorRT build,
timing loops, drift reference and environment record are the same code as in the core layer.

  python sliceb.py check                          environment, GPU, weights, TensorRT test build (run first)
  python sliceb.py run   --out DIR [--configs a,b] [--builds N]   orchestrator (resumable; run in background)
  python sliceb.py watch DIR                      progress bar + ETA (Ctrl+C leaves the job running)
  python sliceb.py status DIR                     one-shot summary
"""
import json, os, random, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                                                    # noqa: E402

HERE = os.path.abspath(__file__)
PY = os.environ.get("PV_PY") or sys.executable
# pre-registered Slice B set (9) + YOLO11l, added before any Slice B data so that both D2 pairs
# (c1pos / SAFEW-288 vs YOLO11m / YOLO11l are D2; here c1pos vs YOLO11m and YOLO11l) can be read on each GPU;
# reported as exploratory (deviation log)
PREREG = ["base640", "p2_640", "c1pos", "Bplain", "y11m", "p2_832", "base960", "p2_960", "base1120"]
EXPLORATORY = ["y11l"]
B_BATCHES = (1, 8)
# usable CPU cores differ between hosts; core.py used 14 threads on the 4090 host
core.THREADS = min(14, len(os.sched_getaffinity(0)))
EXPECT = {"torch": "2.13.0+cu132", "tensorrt": "11.4.0.106", "ultralytics": "8.4.143", "python": "3.12"}


# ----------------------------------------------------------------------------- plan (same structure as core)
def make_plan_b(cfgs, nbuild, shuffle_cfgs):
    rng = random.Random(f"{core.SEED}-sliceb")
    order = list(cfgs)
    if shuffle_cfgs:
        rng.shuffle(order)
    plan = []
    for cfg in order:
        plan.append(("prep", cfg))
        builds = [("build", cfg, "fp16", b, k) for b in B_BATCHES for k in range(1, nbuild + 1)]
        rng.shuffle(builds)
        plan += builds
        plan.append(("ref", cfg, "start"))
        p1 = [("time", cfg, b, fam, 1) for b in B_BATCHES for fam in ("torch", "trt")]
        rng.shuffle(p1)
        plan += p1
        plan.append(("ref", cfg, "mid"))
        p2 = [("time", cfg, b, fam, 2) for b in B_BATCHES for fam in ("torch", "trt")]
        rng.shuffle(p2)
        plan += p2
        plan.append(("ref", cfg, "end"))
        plan.append(("cleanup", cfg))
    return plan


def cmd_for_b(out, s):
    k = s[0]
    if k == "prep":
        return [PY, HERE, "_prepb", out, s[1]]
    if k == "build":
        return [PY, HERE, "_build", out, s[1], s[2], str(s[3]), str(s[4])]
    if k == "time":
        return [PY, HERE, "_time", out, s[1], str(s[2]), s[3], str(s[4])]
    if k == "ref":
        return [PY, HERE, "_time", out, s[1], "1", "ref", s[2]]
    return None


core.make_plan = make_plan_b
core.cmd_for = cmd_for_b
core.DEFAULT_S.update({"prep": 15, "build": 170, "time_torch": 40, "time_trt": 30, "ref": 8})


# ----------------------------------------------------------------------------- _prepb: FP16 export at batch 1 and 8
def w_prepb(out, cfg):
    """Same export as core.w_prep (fused trained weights, Detect anchors cached for the shape, opset 17,
    TorchScript exporter), restricted to the FP16 graphs Slice B builds."""
    import copy, torch
    name, run_name, s = core.CFG[cfg]
    d = core.cdir(out, cfg)
    os.makedirs(os.path.join(d, "onnx"), exist_ok=True)
    m = core.load_model(run_name)
    meta = {"cfg": cfg, "run": run_name, "imgsz": s, "detect_idx": len(m.model) - 1,
            "params": sum(p.numel() for p in m.parameters()), "torch": torch.__version__,
            "exploratory": cfg in EXPLORATORY}
    m16 = copy.deepcopy(m).half()
    w16 = core.wrap(m16)
    del m
    for b in B_BATCHES:
        x = torch.zeros(b, 3, s, s, device="cuda").half()
        with torch.no_grad():
            w16(x)
        p = os.path.join(d, "onnx", f"fp16_b{b}.onnx")
        if not os.path.exists(p):
            torch.onnx.export(w16, (x,), p + ".tmp", opset_version=17, input_names=["images"],
                              output_names=["out"], dynamo=False)
            os.replace(p + ".tmp", p)
        torch.cuda.empty_cache()
    core.jdump(meta, os.path.join(d, "meta.json"))
    print("PREP_OK", meta)


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
        print(f"  [{'OK' if good else '!!'}] {tag:12s} {msg}")

    print("Slice B environment check\n")
    print("  GPU        ", sh("nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit "
                              "--format=csv,noheader"))
    print("  CPU        ", sh("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'"),
          f"| usable cores {len(os.sched_getaffinity(0))} -> threads {core.THREADS}")
    other = sh("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader")
    line("GPU idle", not other, other or "no other GPU processes")
    import platform
    line("python", platform.python_version().startswith(EXPECT["python"]), platform.python_version())
    try:
        import torch
        cap = torch.cuda.get_device_capability(0)
        a = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        float((a @ a).sum())
        line("torch", torch.__version__ == EXPECT["torch"],
             f"{torch.__version__} (4090 run: {EXPECT['torch']}), CUDA works, sm_{cap[0]}{cap[1]}, "
             f"{torch.cuda.get_device_name(0)}")
        line("sm support", f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list() or
             any(x.startswith("compute_") for x in torch.cuda.get_arch_list()),
             "torch build includes this GPU: " + " ".join(torch.cuda.get_arch_list()))
    except Exception as ex:
        line("torch", False, repr(ex)[:300])
        print("\nCHECK FAILED")
        return
    try:
        import ultralytics, torchvision
        line("ultralytics", ultralytics.__version__ == EXPECT["ultralytics"],
             f"{ultralytics.__version__}, torchvision {torchvision.__version__}")
    except Exception as ex:
        line("ultralytics", False, repr(ex)[:300])
    # weights and the custom modules they need
    for cfg in PREREG + EXPLORATORY:
        run_name = core.CFG[cfg][1]
        p = os.path.join(core.RUNS, run_name, "weights", "best.pt")
        if not os.path.exists(p):
            line(cfg, False, f"missing {p}")
            continue
        try:
            m = core.load_model(run_name)
            n = sum(q.numel() for q in m.parameters())
            line(cfg, True, f"{run_name}  {n / 1e6:.2f} M params")
            del m
            torch.cuda.empty_cache()
        except Exception as ex:
            line(cfg, False, f"cannot load {p}: {repr(ex)[:200]}")
    # TensorRT: a tiny FP16 network built and run on this GPU
    try:
        trt = core.import_trt()
        import tempfile
        net_t = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.SiLU()).cuda().half().eval()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "t.onnx")
            torch.onnx.export(net_t, (torch.zeros(1, 3, 64, 64, device="cuda").half(),), p, opset_version=17,
                              input_names=["images"], output_names=["out"], dynamo=False)
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser = trt.OnnxParser(net, logger)
            if not parser.parse_from_file(p):
                raise RuntimeError(parser.get_error(0))
            blob = builder.build_serialized_network(net, builder.create_builder_config())
            if blob is None:
                raise RuntimeError("build returned None")
            eng = trt.Runtime(logger).deserialize_cuda_engine(blob)
            ctx = eng.create_execution_context()
            xi = torch.ones(1, 3, 64, 64, device="cuda").half()
            yo = torch.empty(1, 16, 64, 64, device="cuda").half()
            ctx.set_tensor_address("images", xi.data_ptr())
            ctx.set_tensor_address("out", yo.data_ptr())
            st = torch.cuda.Stream()
            ctx.execute_async_v3(st.cuda_stream)
            st.synchronize()
            ref = net_t(xi)
            err = float((yo - ref).abs().max())
        line("tensorrt", trt.__version__ == EXPECT["tensorrt"] and err < 1e-2,
             f"{trt.__version__} (4090 run: {EXPECT['tensorrt']}), test engine built and run, max diff {err:.1e}")
    except Exception as ex:
        line("tensorrt", False, repr(ex)[:300])
    try:
        import onnx
        line("onnx", True, onnx.__version__)
    except Exception as ex:
        line("onnx", False, repr(ex)[:200])
    free = shutil.disk_usage("/workspace" if os.path.isdir("/workspace") else "/").free / 2**30
    line("disk", free > 15, f"{free:.1f} GB free (Slice B needs < 5 GB; engines are removed after each config)")
    print("\nCHECK PASSED" if ok else "\nCHECK FOUND PROBLEMS (lines marked !!) - send this output")


# ----------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        role, a = sys.argv[1], sys.argv[2:]
        if role == "_prepb":
            w_prepb(a[0], a[1])
        elif role == "_build":
            core.w_build(a[0], a[1], a[2], int(a[3]), int(a[4]))
        elif role == "_time":
            core.w_time(a[0], a[1], int(a[2]), a[3], a[4])
        return
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check")
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--configs", default="")
    r.add_argument("--builds", type=int, default=3)
    for n in ("watch", "status"):
        sp.add_parser(n).add_argument("out")
    args = ap.parse_args()
    if args.cmd == "check":
        check()
    elif args.cmd == "run":
        out = os.path.abspath(args.out)
        os.makedirs(out, exist_ok=True)
        cfgs = args.configs.split(",") if args.configs else PREREG + EXPLORATORY
        meta_path = os.path.join(out, "sliceb_meta.json")
        if not os.path.exists(meta_path):
            core.jdump({"layer": "Slice B", "configs": cfgs, "prereg": PREREG, "exploratory": EXPLORATORY,
                        "batches": list(B_BATCHES), "builds": args.builds, "threads": core.THREADS,
                        "created": time.strftime("%F %T"),
                        "gpu": sh("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader"),
                        "cpu": sh("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'")}, meta_path)
        args.out = out
        _run(args, cfgs)
    elif args.cmd == "watch":
        try:
            core.show(os.path.abspath(args.out))
        except KeyboardInterrupt:
            print("\n(left the viewer; the job keeps running)")
    else:
        core.show(os.path.abspath(args.out), once=True)


def _run(args, cfgs):
    """core.run with the Slice B plan; the full set is shuffled with a fixed seed, an explicit --configs list is not."""
    full = cfgs == PREREG + EXPLORATORY
    plan_path = os.path.join(args.out, "plan.json")
    if not os.path.exists(plan_path):
        core.jdump([list(x) for x in make_plan_b(cfgs, args.builds, shuffle_cfgs=full)], plan_path)
        core.jdump({"configs": cfgs, "builds": args.builds, "created": time.strftime("%F %T"), "seed": core.SEED,
                    "share_ab": False}, os.path.join(args.out, "run_meta.json"))
    core.run(type("A", (), {"out": args.out, "configs": ",".join(cfgs), "builds": args.builds, "share_ab": False})())


if __name__ == "__main__":
    main()
