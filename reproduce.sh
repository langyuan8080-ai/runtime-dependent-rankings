#!/usr/bin/env bash
# Recompute every analysis output and figure of the paper from results/ (CPU only, no GPU or dataset needed)
# and compare the recomputed tables with analysis/reference/, the files used for the paper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/analysis/output"
mkdir -p "$OUT/figs"
cd "$OUT"
for d in full_res stageP extra_res sc_full sliceb_3080ti sliceb_5090 appx; do ln -sfn "$ROOT/results/$d" "$d"; done
A="$ROOT/analysis"
echo "[1/8] RQ1 pair comparisons (two-stage bootstrap, 20,000 resamples)";  python "$A/rq1.py"
echo "[2/8] RQ2 host share per cell";                                       python "$A/rq2.py"
echo "[3/8] RQ3 Slice A crossover batch sizes";                             python "$A/rq3.py"
echo "[4/8] RQ4 accuracy and NMS share";                                    python "$A/rq4.py"
echo "[5/8] H1d Pareto fronts: original rule, corrected rule, without single-seed models"
python "$A/h1d.py"; python "$A/h1d_v2.py"; python "$A/h1d_noY.py"
echo "[6/8] Slice B pair comparisons (RTX 4090 / 3080 Ti / 5090)"
python "$A/sliceb_an.py" 4090=full_res/latency.csv 3080ti=sliceb_3080ti/latency.csv 5090=sliceb_5090/latency.csv
echo "[7/8] Overlap model, Eq. (1)";                                        python "$A/latmodel.py" | tee latmodel.txt
echo "[8/8] Figures";                                                       python "$ROOT/figures/figs.py"
echo
echo "Comparison with analysis/reference/:"
python - "$ROOT/analysis/reference" <<'PY'
import sys, os, pandas as pd, numpy as np
ref = sys.argv[1]; bad = 0
for f in sorted(os.listdir(ref)):
    a, b = pd.read_csv(os.path.join(ref, f)), pd.read_csv(f)
    same = a.shape == b.shape and all(
        np.allclose(a[c], b[c], rtol=1e-9, atol=1e-12, equal_nan=True) if pd.api.types.is_numeric_dtype(a[c])
        else a[c].astype(str).equals(b[c].astype(str)) for c in a.columns)
    bad += not same
    print(f"  {'identical' if same else 'DIFFERS  '}  {f}")
print("All recomputed tables are identical to the reference." if not bad else f"{bad} file(s) differ.")
PY
