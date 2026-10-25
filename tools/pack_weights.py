#!/usr/bin/env python3
"""Pack the best.pt files of the 68 runs used in the paper (tools/weights_runs.txt) from a local weights archive
into weights_best_68.tar with the layout runs/detect/<run>/weights/best.pt, and write SHA256SUMS_weights.txt.

  python tools/pack_weights.py <archive.tar | folder> [--out weights_best_68.tar]
"""
import hashlib, io, os, sys, tarfile

here = os.path.dirname(os.path.abspath(__file__))
runs = [r.strip() for r in open(os.path.join(here, "weights_runs.txt")) if r.strip()]
src = sys.argv[1]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "weights_best_68.tar"


def run_of(path):
    p = path.replace("\\", "/").split("/")
    if len(p) >= 3 and p[-1] == "best.pt" and p[-2] == "weights" and p[-3] in runs:
        return p[-3]


found = {}
if os.path.isdir(src):
    for root, _, files in os.walk(src):
        for f in files:
            full = os.path.join(root, f); r = run_of(full)
            if r and r not in found:
                found[r] = ("dir", full)
else:
    tin = tarfile.open(src)
    for m in tin.getmembers():
        r = run_of(m.name)
        if m.isfile() and r and r not in found:
            found[r] = ("tar", m)
missing = [r for r in runs if r not in found]
if missing:
    sys.exit(f"missing best.pt for {len(missing)} runs: {', '.join(missing)}")
sums = []
with tarfile.open(out, "w") as tout:
    for r in runs:
        kind, obj = found[r]
        data = open(obj, "rb").read() if kind == "dir" else tin.extractfile(obj).read()
        name = f"runs/detect/{r}/weights/best.pt"
        info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0
        tout.addfile(info, io.BytesIO(data))
        sums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
open("SHA256SUMS_weights.txt", "w").write("\n".join(sums) + "\n")
print(f"wrote {out} ({len(runs)} runs, {os.path.getsize(out) / 1e6:.0f} MB) and SHA256SUMS_weights.txt")
