#!/usr/bin/env python3
"""
slicec.py — Paper 2, Slice C: accuracy under every runtime/precision (RQ4: H4a, H4b; accuracy axis of H1d) and
batch-1 post-processing share (H4c). Pre-registration Section 7.

  python slicec.py run   --out DIR [--smoke]      orchestrator (resumable; run in background)
  python slicec.py watch DIR                       progress bars + ETA (Ctrl+C leaves the job running)
  python slicec.py status DIR                      one-shot summary

Stage R  reference check: every seed's best.pt, eager FP32, evaluated exactly as v52 (Ultralytics val, rectangular
         batches of 8, max_det 317) on val and test-dev; must reproduce the v52 values within 0.05 points.
         A gate step stops the run if any run with a v52 reference misses it.
Stage C  per seed: static square-letterbox evaluation (TensorRT engines need a fixed input) of
         eager FP32 | ONNX Runtime FP32 | TensorRT FP32 | FP16 | INT8 (a) | INT8 (b), batch 8, the same
         Ultralytics validator (only the forward pass is replaced), mAP@0.5:0.95 / mAP@0.5 from Ultralytics, AP_small
         from pycocotools on the saved predictions (as v52 E2), numerical sanity check per engine on 32 val images.
         INT8: entropy calibration on the fixed 512-image list for every seed (seed 0 of the core configurations
         reuses the core-layer calibration), scheme b from scheme a, batch-8 graph = rebatched batch-1 graph.
         Seed 0 also gets the batch-1 forward vs NMS timing (eager, TensorRT FP16 + CUDA Graph) on 200 val images.
Requires core.py in the same directory.
"""
import argparse, collections, csv, glob, json, os, random, re, shutil, subprocess, sys, time, traceback, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                                                    # noqa: E402
from core import PV_PY, QV_PY, jdump, jload, cdir, bar, hms, THREADS, SEED  # noqa: E402

HERE = os.path.abspath(__file__)
WS = "/workspace"
FULL = os.environ.get("FULL_DIR", "/workspace/paper2_core/full")
REF_DIR = os.environ.get("REF_DIR", "/workspace/recheck_out")
EVAL_BATCH = 8
MAX_DET = 317
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
PATHS = ["eager_sq", "ort_fp32", "trt_fp32", "trt_fp16", "trt_int8a", "trt_int8b"]
MIN_FREE_GB = 4

# configuration -> (input, seed runs); seed 0 first. The first 19 are the core-layer configurations.
CONFIG_RUNS = {
    "base640": (640, ["baseline_scratch", "baseline_s1", "baseline_s2"]),
    "p2_640": (640, ["stage6_p2fix_s0", "stage6_p2fix_s1", "stage6_p2fix_s2"]),
    "afterA": (640, ["safe_p2_yolo11s", "safep2_s1", "safep2_s2", "sci_after_s3", "sci_after_s4"]),
    "c1bn": (640, ["stage5_c1bn_s0", "stage5_c1bn_s1", "stage5_c1bn_s2"]),
    "c1pos": (640, ["stage3_c1pos", "stage4_c1pos_s1", "stage4_c1pos_s2", "sci_c1pos_s3", "sci_c1pos_s4"]),
    "Awide": (640, ["stage7_Awide_s0", "stage7_Awide_s1", "stage7_Awide_s2", "sci_Awide_s3", "sci_Awide_s4"]),
    "Bnarrow": (640, ["sci_Bnarrow_s0", "sci_Bnarrow_s1", "sci_Bnarrow_s2"]),
    "Bplain": (640, ["stage7_Bplain_s0", "stage7_Bplain_s1", "stage7_Bplain_s2", "sci_Bplain_s3", "sci_Bplain_s4"]),
    "Bg4": (640, ["sci_Bg4_s0", "sci_Bg4_s1", "sci_Bg4_s2"]),
    "seg1": (640, ["cap_se_g1_s0", "cap_se_g1_s1", "cap_se_g1_s2"]),
    "seg4": (640, ["cap_se_g4_s0", "cap_se_g4_s1", "cap_se_g4_s2"]),
    "safew288": (640, ["cap_safew288_s0", "cap_safew288_s1", "cap_safew288_s2"]),
    "y11m": (640, ["yolo11m_s0"]),
    "y11l": (640, ["yolo11l_s0"]),
    "base960": (960, ["hr_base960_s0", "hr_base960_s1", "hr_base960_s2"]),
    "base1024": (1024, ["hr_base1024_s0", "hr_base1024_s1", "hr_base1024_s2"]),
    "base1120": (1120, [f"hr_base1120_s{i}" for i in range(5)]),
    "p2_832": (832, ["hr_p2_832_s0", "hr_p2_832_s1", "hr_p2_832_s2"]),
    "p2_960": (960, [f"hr_p2_960_s{i}" for i in range(5)]),
    "base960b16": (960, ["sci_base960_s0", "sci_base960_s1", "sci_base960_s2"]),
}
RUN = {r: (cfg, s, i) for cfg, (s, rs) in CONFIG_RUNS.items() for i, r in enumerate(rs)}   # run -> (cfg, input, seed)
core.CFG.update({r: (r, r, s) for r, (cfg, s, i) in RUN.items()})                        # core workers key by run
A_KEYS = ["ts", "cfg", "run", "seed", "imgsz", "path", "split", "map5095", "map50", "P", "R", "status", "seconds"]
C_KEYS = ["ts", "cfg", "run", "seed", "path", "split", "coco_AP", "coco_AP50", "AP_s", "AP_m", "AP_l", "dets_per_img",
          "status"]
R_KEYS = ["ts", "cfg", "run", "seed", "imgsz", "split", "map5095", "map50", "ref_map5095", "ref_source", "diff_points",
          "check"]
S_KEYS = ["ts", "cfg", "run", "seed", "path", "n_images", "max_abs", "mean_abs", "max_abs_scores", "out_range"]
L_KEYS = ["ts", "cfg", "run", "imgsz", "unit", "conf", "n_images", "fwd_ms", "nms_ms", "e2e_ms", "nms_share",
          "nms_share_median_img", "dets_per_img", "nms_impl"]


def write_rows(path, keys, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def data_yamls(out):
    """v52 used /workspace/visdrone.yaml (val) and /workspace/visdrone_testdev.yaml (split=test); fall back to our own."""
    v, t = os.path.join(WS, "visdrone.yaml"), os.path.join(WS, "visdrone_testdev.yaml")
    if os.path.exists(v) and os.path.exists(t):
        return {"val": v, "test": t}
    own = os.path.join(out, "visdrone_c.yaml")
    with open(own, "w") as f:
        f.write(f"path: {WS}/data\ntrain: VisDrone2019-DET-train/images\nval: VisDrone2019-DET-val/images\n"
                f"test: VisDrone2019-DET-test-dev/images\nnames:\n" +
                "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return {"val": own, "test": own}


SPLIT_DIR = {"val": os.path.join(WS, "data", "VisDrone2019-DET-val"),
             "test": os.path.join(WS, "data", "VisDrone2019-DET-test-dev")}


# ----------------------------------------------------------------------------- model loading
def load_yolo(run):
    os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "0")
    import torch.nn as nn
    from ultralytics import YOLO
    y = YOLO(os.path.join(core.RUNS, run, "weights", "best.pt"))
    for mod in y.model.modules():
        if type(mod).__name__ == "SAFE":
            if not hasattr(mod, "sa"):
                mod.sa = nn.Identity()
            for a, v in (("use_spatial", False), ("gamma_per_channel", False), ("use_multiscale", True),
                         ("use_attention", True), ("use_gamma", True)):
                if not hasattr(mod, a):
                    setattr(mod, a, v)
    return y


def make_backend(out, run, path, batch=EVAL_BATCH):
    """fn(x: float32 CUDA tensor (n, 3, s, s), n <= batch) -> float32 CUDA tensor (n, 4 + nc, anchors)."""
    import torch
    od = os.path.join(cdir(out, run), "onnx")
    if path == "ort_fp32":
        import numpy as np, onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        sess = ort.InferenceSession(os.path.join(od, f"fp32_b{batch}.onnx"), so, providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError("ORT fell back to CPU")

        def fn(x):
            n = x.shape[0]
            xp = torch.cat([x, x.new_zeros(batch - n, *x.shape[1:])]) if n < batch else x
            y = sess.run(None, {"images": xp.float().cpu().numpy()})[0]
            return torch.from_numpy(y[:n]).to(x.device).float()
        return fn
    prec = path.split("_")[1]
    trt = core.import_trt()
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    TDT = {trt.float32: torch.float32, trt.float16: torch.float16}
    with open(os.path.join(cdir(out, run), "engines", f"{prec}_b{batch}_k1.plan"), "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        t = eng.get_tensor_name(i)
        bufs[t] = torch.zeros(tuple(eng.get_tensor_shape(t)), dtype=TDT[eng.get_tensor_dtype(t)], device="cuda")
        ctx.set_tensor_address(t, bufs[t].data_ptr())
    stream = torch.cuda.Stream()

    def fn(x):
        n = x.shape[0]
        inb = bufs["images"]
        inb.zero_()
        inb[:n].copy_(x.to(inb.dtype))
        torch.cuda.current_stream().synchronize()
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return bufs["out"][:n].float().clone()
    fn._keep = (eng, ctx, bufs, stream)
    return fn


# ----------------------------------------------------------------------------- evaluation helpers
def patch_eval_json():
    """Keep Ultralytics' own metrics: save_json is used only to obtain the predictions file for pycocotools AP_small."""
    from ultralytics.models.yolo.detect import val as dv
    dv.DetectionValidator.eval_json = lambda self, stats: stats


def coco_gt(split):
    """Ground truth from the YOLO labels in original-image pixels (as v52 E2); image ids are the file stems."""
    from PIL import Image
    d = SPLIT_DIR[split]
    imgs, anns, aid = [], [], 1
    files = sorted(p for p in glob.glob(os.path.join(d, "images", "*")) if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png"))
    for iid, p in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        with Image.open(p) as im:
            W, H = im.size
        imgs.append({"id": iid, "file_name": stem, "width": W, "height": H})
        lp = os.path.join(d, "labels", stem + ".txt")
        if os.path.isfile(lp):
            for line in open(lp):
                v = line.split()
                if len(v) < 5:
                    continue
                c, xc, yc, w, h = int(v[0]), *map(float, v[1:5])
                bw, bh = w * W, h * H
                anns.append({"id": aid, "image_id": iid, "category_id": c + 1, "iscrowd": 0, "area": bw * bh,
                             "bbox": [xc * W - bw / 2, yc * H - bh / 2, bw, bh]})
                aid += 1
    return {"images": imgs, "annotations": anns, "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}


def coco_metrics(out, split, pred_json):
    import contextlib, io, numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gp = os.path.join(out, f"gt_{split}.json")
    if not os.path.exists(gp):
        jdump(coco_gt(split), gp)
    gt = jload(gp)
    sid = {im["file_name"]: im["id"] for im in gt["images"]}
    dets = [{"image_id": sid[str(d["image_id"])], "category_id": d["category_id"], "bbox": d["bbox"], "score": d["score"]}
            for d in jload(pred_json, []) if str(d["image_id"]) in sid]
    if not dets:
        return {"status": "no_detections"}
    C = COCO()
    C.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        C.createIndex()
        D = C.loadRes(dets)
        E = COCOeval(C, D, "bbox")
        E.params.maxDets = [1, 10, MAX_DET]
        E.evaluate()
        E.accumulate()
    P = E.eval["precision"]

    def ap(iou=None, area=0):
        t = slice(None) if iou is None else int(np.where(np.isclose(E.params.iouThrs, iou))[0][0])
        x = P[t, :, :, area, -1]
        x = x[x > -1]
        return round(float(x.mean()), 5) if x.size else float("nan")
    return {"coco_AP": ap(), "coco_AP50": ap(0.5), "AP_s": ap(area=1), "AP_m": ap(area=2), "AP_l": ap(area=3),
            "dets_per_img": round(len(dets) / len(gt["images"]), 1)}


def letterboxed(paths, s):
    import cv2, numpy as np, torch
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=(s, s), auto=False, scaleup=False)
    ims = [lb(image=cv2.imread(p)) for p in paths]
    x = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in ims]).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


def val_images(n):
    return sorted(glob.glob(os.path.join(SPLIT_DIR["val"], "images", "*.jpg")))[:n]


# ----------------------------------------------------------------------------- workers
def w_ref(out, run):
    """v52 protocol: Ultralytics val, rectangular batches of 8, max_det 317, eager FP32 (fused), val and test-dev."""
    cfg, s, seed = RUN[run]
    ys = data_yamls(out)
    rows = []
    for split in ("val", "test"):
        y = load_yolo(run)
        r = y.val(data=ys[split], split=split, imgsz=s, batch=8, max_det=MAX_DET, device=0, verbose=False,
                  plots=False, project=os.path.join(out, "valruns"), name=f"ref_{run}_{split}", exist_ok=True)
        m, m50 = float(r.box.map), float(r.box.map50)
        ref, src = reference(run, split, s)
        diff = None if ref is None else round((m - ref) * 100, 3)
        rows.append({"ts": time.strftime("%F %T"), "cfg": cfg, "run": run, "seed": seed, "imgsz": s, "split": split,
                     "map5095": round(m, 5), "map50": round(m50, 5), "ref_map5095": ref, "ref_source": src,
                     "diff_points": diff, "check": "no_reference" if ref is None else ("pass" if abs(diff) <= 0.05 else "FAIL")})
        del y
    write_rows(os.path.join(out, "reference.csv"), R_KEYS, rows)
    print("REF_OK", rows)


def reference(run, split, s):
    """v52 standalone values: recheck_out/v4_E1_map.csv (640 runs, val and test) and hr_testdev_results.csv (ExpH, test)."""
    p = os.path.join(REF_DIR, "v4_E1_map.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r.get("run") == run and r.get("weights", "best.pt") == "best.pt" and r.get("split") == split \
                    and r.get("status", "ok") == "ok" and int(float(r.get("imgsz") or 640)) == s and r.get("mAP50-95"):
                return float(r["mAP50-95"]), "v4_E1_map.csv"
    p = os.path.join(REF_DIR, "hr_testdev_results.csv")
    if split == "test" and os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r.get("run") == run and int(float(r.get("imgsz", 0))) == s:
                return float(r["mAP50_95"]) / 100, "hr_testdev_results.csv"
    return None, ""


def w_prep(out, run):
    """Export FP32/FP16 graphs; seed 0 of a core configuration reuses the core-layer INT8 calibration when the
    re-export is byte-identical to the core-layer export."""
    import hashlib
    core.w_prep(out, run)
    cfg, s, seed = RUN[run]
    od = os.path.join(cdir(out, run), "onnx")
    src = os.path.join(cdir(FULL, cfg), "onnx")
    q = jload(os.path.join(cdir(FULL, cfg), "quant.json"), {}) or {}
    reuse = False
    if seed == 0 and core.CFG.get(cfg, (None, None))[1] == run and q.get("fp32_md5"):
        md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
        if all(md5(os.path.join(od, f"{k}.onnx")) == v for k, v in q["fp32_md5"].items()):
            for p in ("int8a", "int8b"):
                shutil.copyfile(os.path.join(src, f"{p}_b1.onnx"), os.path.join(od, f"{p}_b1.onnx"))
            keep = {k: v for k, v in q.items() if k in ("modelopt", "procedure", "int8a_b1", "int8b_b1")}
            jdump(keep, os.path.join(cdir(out, run), "quant.json"))
            reuse = True
    jdump({"reuse_core_calibration": reuse}, os.path.join(cdir(out, run), "prep_c.json"))
    print("PREPC_OK", run, "reuse core calibration:", reuse)


def w_eval(out, run, path):
    import numpy as np, torch
    cfg, s, seed = RUN[run]
    patch_eval_json()
    ys = data_yamls(out)
    rows, t0 = [], time.time()
    eager = load_yolo(run).model.fuse(verbose=False).cuda().eval() if path != "eager_sq" else None
    fn = make_backend(out, run, path) if path != "eager_sq" else None
    if fn is not None:                                   # numerical sanity check, 32 real validation images
        x = letterboxed(val_images(32), s)
        d, ds = [], []
        with torch.no_grad():
            for i in range(0, 32, EVAL_BATCH):
                ye = eager(x[i:i + EVAL_BATCH])
                ye = (ye[0] if isinstance(ye, (list, tuple)) else ye).float()
                yr = fn(x[i:i + EVAL_BATCH])
                d.append((yr - ye).abs())
                ds.append((yr[:, 4:] - ye[:, 4:]).abs())
        d, ds = torch.cat(d), torch.cat(ds)
        write_rows(os.path.join(out, "sanity.csv"), S_KEYS, [{
            "ts": time.strftime("%F %T"), "cfg": cfg, "run": run, "seed": seed, "path": path, "n_images": 32,
            "max_abs": round(float(d.max()), 6), "mean_abs": round(float(d.mean()), 6),
            "max_abs_scores": round(float(ds.max()), 6), "out_range": round(float(ye.abs().max()), 3)}])
        del eager
        torch.cuda.empty_cache()
    for split in ("val", "test"):
        y = load_yolo(run)
        if fn is not None:
            y.model.forward = types.MethodType(lambda self, x, *a, **k: fn(x), y.model)
        name = f"{path}_{run}_{split}"
        r = y.val(data=ys[split], split=split, imgsz=s, batch=EVAL_BATCH, max_det=MAX_DET, rect=False, save_json=True,
                  device=0, verbose=False, plots=False, project=os.path.join(out, "valruns"), name=name, exist_ok=True)
        row = {"ts": time.strftime("%F %T"), "cfg": cfg, "run": run, "seed": seed, "imgsz": s, "path": path,
               "split": split, "map5095": round(float(r.box.map), 5), "map50": round(float(r.box.map50), 5),
               "P": round(float(r.box.mp), 5), "R": round(float(r.box.mr), 5), "status": "ok"}
        pj = os.path.join(out, "valruns", name, "predictions.json")
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        if os.path.exists(pj):        # pycocotools (AP_small) runs in a background CPU worker; raw file deleted there
            os.makedirs(os.path.join(out, "pending"), exist_ok=True)
            os.replace(pj, os.path.join(out, "pending", f"{run}__{path}__{split}.json"))
        else:
            row["status"] = "no_predictions_file"
        del y
    write_rows(os.path.join(out, "accuracy.csv"), A_KEYS, rows)
    print("EVAL_OK", run, path, [(r["split"], r["map5095"], r.get("AP_s")) for r in rows])


def w_lat(out, run):
    """H4c: batch-1 forward vs NMS on 200 letterboxed validation images; eager FP32 and TensorRT FP16 + CUDA Graph."""
    import statistics as st, torch
    import torchvision  # noqa: F401  Ultralytics uses torchvision NMS only when torchvision is imported (its predictor does)
    from ultralytics.utils.nms import non_max_suppression as nms
    torch.set_num_threads(THREADS)
    cfg, s, seed = RUN[run]
    core.env_record(out, run, "lat")
    x = letterboxed(val_images(200), s)
    m = core.load_model(run)
    units = {"eager": lambda xi: m(xi)}
    fn = make_backend(out, run, "trt_fp16", batch=1)
    eng, ctx, bufs, stream = fn._keep
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        with torch.cuda.graph(g, stream=stream):
            ctx.execute_async_v3(stream.cuda_stream)

    def trt_unit(xi):
        bufs["images"].copy_(xi.to(bufs["images"].dtype))
        g.replay()
        return bufs["out"].float()
    units["trt_fp16_graph"] = trt_unit
    rows = []
    with torch.no_grad():
        for u, f in units.items():
            for _ in range(core.WARM):
                y0 = f(x[:1])
                y0 = y0[0] if isinstance(y0, (list, tuple)) else y0
                for conf in (0.25, 0.001):
                    nms(y0, conf_thres=conf, iou_thres=0.7, max_det=MAX_DET)
            torch.cuda.synchronize()
            for conf in (0.25, 0.001):
                tf, tn, sh, nd = [], [], [], []
                for i in range(len(x)):
                    xi = x[i:i + 1]
                    t0 = time.perf_counter()
                    y = f(xi)
                    y = y[0] if isinstance(y, (list, tuple)) else y
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    det = nms(y, conf_thres=conf, iou_thres=0.7, max_det=MAX_DET)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    tf.append((t1 - t0) * 1e3)
                    tn.append((t2 - t1) * 1e3)
                    sh.append((t2 - t1) / (t2 - t0))
                    nd.append(len(det[0]))
                fm, nm = st.median(tf), st.median(tn)
                rows.append({"ts": time.strftime("%F %T"), "cfg": cfg, "run": run, "imgsz": s, "unit": u, "conf": conf,
                             "n_images": len(x), "fwd_ms": round(fm, 4), "nms_ms": round(nm, 4), "e2e_ms": round(fm + nm, 4),
                             "nms_share": round(nm / (fm + nm), 4), "nms_share_median_img": round(st.median(sh), 4),
                             "dets_per_img": round(st.mean(nd), 1), "nms_impl": "torchvision"})
    write_rows(os.path.join(out, "latency_C.csv"), L_KEYS, rows)
    print("LAT_OK", rows)


def w_quant(out, run):
    core.w_quant(out, run)


def w_coco(out):
    """Background CPU worker: pycocotools metrics (AP, AP_small, ...) for every predictions file the evaluations hand
    over in pending/; appends to coco.csv and deletes the raw file. Exits when STOP_COCO exists and nothing is pending."""
    pend = os.path.join(out, "pending")
    os.makedirs(pend, exist_ok=True)
    while True:
        files = sorted(f for f in os.listdir(pend) if f.endswith(".json"))
        if not files:
            if os.path.exists(os.path.join(out, "STOP_COCO")):
                return
            time.sleep(10)
            continue
        for f in files:
            run_, path, split = f[:-5].split("__")
            cfg, s, seed = RUN[run_]
            try:
                m = coco_metrics(out, split, os.path.join(pend, f))
                m.setdefault("status", "ok")
            except Exception as ex:
                traceback.print_exc()
                m = {"status": f"error: {type(ex).__name__}"}
            write_rows(os.path.join(out, "coco.csv"), C_KEYS, [dict(ts=time.strftime("%F %T"), cfg=cfg, run=run_,
                                                                     seed=seed, path=path, split=split, **m)])
            os.remove(os.path.join(pend, f))
            print("COCO_OK", f, m, flush=True)


# ----------------------------------------------------------------------------- plan
def make_plan(smoke=False):
    rng = random.Random(SEED + 3)
    cfgs = list(CONFIG_RUNS)
    if smoke:
        sel = {"base640": ["baseline_scratch", "baseline_s1"]}
    else:
        rng.shuffle(cfgs)
        sel = {c: CONFIG_RUNS[c][1] for c in cfgs}
    plan = [("ref", r) for c in sel for r in sel[c]] + [("gate",)]
    for c, runs in sel.items():
        for r in runs:
            seed = RUN[r][2]
            plan += [("prep", r), ("quant", r)]
            bl = [("build", r, p, "8") for p in ("fp32", "fp16", "int8a", "int8b")] + ([("build", r, "fp16", "1")] if seed == 0 else [])
            rng.shuffle(bl)
            ev = [("eval", r, p) for p in PATHS]
            rng.shuffle(ev)
            plan += bl + ev + ([("lat", r)] if seed == 0 else []) + [("clean", r)]
    return [list(map(str, s)) for s in plan]


def sid(s):
    return "|".join(s)


DEFAULT_S = {"ref": 90, "gate": 1, "prep": 60, "quant": 1100, "build": 140, "eval": 90, "lat": 120, "clean": 1}
TIMEOUT = {"quant": 14400, "build": 3600}


# ----------------------------------------------------------------------------- orchestrator
def run(args):
    out = os.path.abspath(args.out)
    for d in ("logs", "cfg", "valruns"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    plan = jload(os.path.join(out, "plan.json"))
    if plan is None:
        plan = make_plan(args.smoke)
        jdump(plan, os.path.join(out, "plan.json"))
        jdump({"created": time.strftime("%F %T"), "smoke": args.smoke, "full": FULL, "ref_dir": REF_DIR, "seed": SEED,
               "yamls": data_yamls(out)}, os.path.join(out, "run_meta.json"))
    state = jload(os.path.join(out, "state.json"), {"done": {}, "failed": {}})
    with open(os.path.join(out, "orchestrator.pid"), "w") as f:
        f.write(str(os.getpid()))
    coco_worker(out, start=True)
    for s in plan:
        k = sid(s)
        if k in state["done"]:
            continue
        while shutil.disk_usage(out).free < MIN_FREE_GB * 2**30:
            progress(out, plan, state, f"PAUSED: less than {MIN_FREE_GB} GB free (next {k})")
            time.sleep(60)
        if s[0] == "lat":                                   # timing step: no background CPU work meanwhile
            wait_pending(out, plan, state, k)
        progress(out, plan, state, k)
        t0 = time.time()
        r = s[1] if len(s) > 1 else None
        if r and s[0] not in ("ref", "prep", "quant") and any(f"{x}|{r}" in state["failed"] for x in ("prep", "quant")):
            ok, msg = False, "skipped: preparation or quantization of this run failed"
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
        if s[0] == "gate" and not ok:
            progress(out, plan, state, "STOPPED at the reference gate: " + msg[:200])
            return
    wait_pending(out, plan, state, "FINISHED (waiting for AP_small computations)")
    open(os.path.join(out, "STOP_COCO"), "w").close()
    progress(out, plan, state, "FINISHED")


def coco_worker(out, start=False):
    pidf = os.path.join(out, "coco_worker.pid")
    try:
        os.kill(int(open(pidf).read()), 0)
        return True
    except Exception:
        if not start:
            return False
    if os.path.exists(os.path.join(out, "STOP_COCO")):
        os.remove(os.path.join(out, "STOP_COCO"))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.Popen([PV_PY, HERE, "_coco", out], stdout=open(os.path.join(out, "logs", "coco.log"), "a"),
                         stderr=subprocess.STDOUT, env=env, start_new_session=True)
    open(pidf, "w").write(str(p.pid))
    return True


def n_pending(out):
    d = os.path.join(out, "pending")
    return len([f for f in os.listdir(d) if f.endswith(".json")]) if os.path.isdir(d) else 0


def wait_pending(out, plan, state, label):
    while n_pending(out):
        coco_worker(out, start=True)
        progress(out, plan, state, f"waiting for {n_pending(out)} AP_small computations before: {label}")
        time.sleep(15)


def sub(out, s, cmd, env_extra=None, timeout=7200):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(THREADS)
    env.update(env_extra or {})
    log = os.path.join(out, "logs", f"{s[1] if len(s) > 1 else 'gate'}.log")
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
    k = s[0]
    if k == "ref":
        return sub(out, s, [PV_PY, HERE, "_ref", out, s[1]])
    if k == "gate":
        rows = list(csv.DictReader(open(os.path.join(out, "reference.csv")))) if os.path.exists(os.path.join(out, "reference.csv")) else []
        bad = [f"{r['run']} {r['split']} {r['diff_points']}" for r in rows if r["check"] == "FAIL"]
        n_ref = sum(r["check"] != "no_reference" for r in rows)
        jdump({"rows": len(rows), "with_reference": n_ref, "failed": bad}, os.path.join(out, "gate.json"))
        if not rows:
            return False, "no reference rows"
        return (not bad), ("reference check failed: " + "; ".join(bad)) if bad else ""
    if k == "prep":
        return sub(out, s, [PV_PY, HERE, "_prep", out, s[1]])
    if k == "quant":
        env = {"QUANT_SHARE_AB": "1"}
        if (jload(os.path.join(out, "run_meta.json"), {}) or {}).get("smoke"):
            env["QUANT_NCALIB"] = "32"                       # smoke test only; the measured run uses all 512 images
        return sub(out, s, [QV_PY, HERE, "_quant", out, s[1]], env, timeout=TIMEOUT["quant"])
    if k == "build":
        return sub(out, s, [PV_PY, core.HERE, "_build", out, s[1], s[2], s[3], "1"], timeout=TIMEOUT["build"])
    if k == "eval":
        return sub(out, s, [PV_PY, HERE, "_eval", out, s[1], s[2]])
    if k == "lat":
        return sub(out, s, [PV_PY, HERE, "_lat", out, s[1]])
    if k == "clean":
        d = cdir(out, s[1])
        shutil.rmtree(os.path.join(d, "engines"), ignore_errors=True)
        od = os.path.join(d, "onnx")
        for f in os.listdir(od) if os.path.isdir(od) else []:
            if not re.fullmatch(r"(fp32|int8a|int8b)_b1\.onnx", f):
                os.remove(os.path.join(od, f))
        return True, ""
    return False, f"unknown step {s}"


def progress(out, plan, state, current):
    done = state["done"]
    avg = collections.defaultdict(list)
    for kk, dt in done.items():
        avg[kk.split("|")[0]].append(dt)
    avg = {kk: sum(v) / len(v) for kk, v in avg.items()}
    eta = sum(avg.get(s[0], DEFAULT_S.get(s[0], 60)) for s in plan if sid(s) not in done)
    blocks = collections.OrderedDict()
    for s in plan:
        b = "reference check" if s[0] in ("ref", "gate") else RUN[s[1]][0]
        d = blocks.setdefault(b, [0, 0])
        d[1] += 1
        d[0] += sid(s) in done
    jdump({"total": len(plan), "done": len(done), "failed": len(state["failed"]), "current": current,
           "eta_s": round(eta), "work_s": round(sum(done.values())), "avg_s": {kk: round(v, 1) for kk, v in avg.items()},
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
            print(f"Paper 2 Slice C   {time.strftime('%F %T')}   (updated {P['updated']})\n")
            print(f"Overall  [{bar(p)}] {p * 100:5.1f}%   {P['done']}/{P['total']} steps   failed {P['failed']}")
            print(f"Work time {hms(P['work_s'])}   ETA {hms(P['eta_s'])}")
            print(f"Current  {P['current']}")
            print(f"Disk     {shutil.disk_usage(out).free / 2**30:.1f} GB free")
            for b, (d, t) in P["blocks"].items():
                if 0 < d < t:
                    print(f"  {b:16s} [{bar(d / t, 20)}] {d}/{t}")
            print(f"\n  blocks finished: {sum(1 for d, t in P['blocks'].values() if d == t)}/{len(P['blocks'])}")
            if P["avg_s"]:
                print("  mean step time (s): " + ", ".join(f"{k} {v}" for k, v in sorted(P["avg_s"].items())))
            cp = os.path.join(out, "coco.csv")
            nc = len(list(csv.DictReader(open(cp)))) if os.path.exists(cp) else 0
            print(f"  AP_small computed: {nc}   pending: {n_pending(out)}   background worker: {'running' if coco_worker(out) else 'not running'}")
            rp = os.path.join(out, "reference.csv")
            if os.path.exists(rp):
                rr = list(csv.DictReader(open(rp)))
                c = collections.Counter(r["check"] for r in rr)
                print(f"  reference check: {dict(c)}")
            fl = os.path.join(out, "failures.csv")
            if os.path.exists(fl):
                rows = list(csv.DictReader(open(fl)))
                if rows:
                    print(f"  last failure: {rows[-1]['step']}  {rows[-1]['error'][:160]}")
            if P["current"] == "FINISHED" or P["current"].startswith("STOPPED"):
                print("\n✓ FINISHED" if P["current"] == "FINISHED" else "\n⚠ " + P["current"])
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


def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        role, a = sys.argv[1], sys.argv[2:]
        {"_ref": lambda: w_ref(a[0], a[1]), "_prep": lambda: w_prep(a[0], a[1]), "_quant": lambda: w_quant(a[0], a[1]),
         "_eval": lambda: w_eval(a[0], a[1], a[2]), "_lat": lambda: w_lat(a[0], a[1]), "_coco": lambda: w_coco(a[0])}[role]()
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
