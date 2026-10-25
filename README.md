# Runtime-dependent rankings of detector design choices

Code, raw measurements and analysis for the paper

> L. Lang, J.-w. Kim. **Does the Deployment Runtime Change Which Detector Design Wins? Host-Bound and Device-Bound Latency Regimes of YOLO11-Based Detectors for UAV Imagery.** Submitted to *Journal of Real-Time Image Processing*.

The study measures 20 YOLO11-based detector configurations (19 distinct for latency) from the companion study [Capacity Before Design](https://github.com/langyuan8080-ai/capacity-before-design) in eleven runtime–precision settings (eager PyTorch, CUDA Graphs, torch.compile, ONNX Runtime, TensorRT FP32/FP16/INT8) at batch sizes 1, 8 and 32 on an RTX 4090, profiles host and device time, restricts compute with CUDA MPS, measures accuracy for every training seed and precision, and repeats a subset on an RTX 3080 Ti and an RTX 5090. The research questions, hypotheses, measurement protocol and decision rules were frozen on 16 October 2026, before the first formal measurement; the protocol and its deviation log are deposited at OSF ([doi:10.17605/OSF.IO/TKJ7B](https://doi.org/10.17605/OSF.IO/TKJ7B)).

Every number, table and figure of the paper can be recomputed from the files in `results/` on a CPU (`bash reproduce.sh`). The scripts that produced those files on the GPUs are in `measure/`. The trained weights and the full run logs are archived at Zenodo ([doi:10.5281/zenodo.23631046](https://doi.org/10.5281/zenodo.23631046)).

## Contents

```
measure/            the measurement scripts exactly as run on the GPU servers
  core.py           core layer: 19 configurations x 11 runtime cells x batch {1, 8, 32}, 3 TensorRT builds (RQ1)
  extra.py          profiling layer (Nsight Systems, RQ2) and Slice A (CUDA MPS 100/50/25 %, RQ3)
  sliceb.py         Slice B: subset on other GPUs (RTX 3080 Ti, RTX 5090)
  slicec.py         Slice C: accuracy of all 68 seeds in six precision paths, NMS share (RQ4)
  appx.py           context measurements (TensorRT BF16, CPU, YOLO26-s) and INT8 layer inspection (Online Resource 4)
analysis/           analysis scripts; reference/ holds the output tables used for the paper
figures/figs.py     Figs. 1-3 of the paper and the study-design diagram of Online Resource 1
tools/              weights_runs.txt (the 68 training runs whose best.pt weights were used), pack_weights.py
results/            raw measurements (one folder per measurement layer, see below)
reproduce.sh        recompute all analysis tables and figures and compare them with analysis/reference/
```

## 1. Reproduce the analysis (CPU, no GPU or dataset needed)

```bash
pip install -r requirements.txt
bash reproduce.sh
```

The script links the result folders into `analysis/output/`, runs the analysis scripts in order, draws the figures into `analysis/output/figs/` and reports for each table whether it is identical to the version in `analysis/reference/`. The bootstrap analyses use fixed random seeds; a full run takes about 15 minutes on one CPU core.

| Paper item | Source data | Script → output |
|---|---|---|
| Section 4.1, H1a–H1c, Online Resource 2 (all pair comparisons, outcome classes) | `results/full_res/latency.csv` | `analysis/rq1.py` → `rq1_pairs.csv` |
| Fig. 1 (D2 latency ratios) | `rq1_pairs.csv` | `figures/figs.py` |
| Section 4.2, H2a–H2c, H3b, Fig. 3a | `results/full_res/latency.csv`, `results/stageP/profile.csv` | `analysis/rq2.py` → `rq2_cells.csv`; `rq1.py` → `rq2_ols.csv` |
| Eq. (1), overlap model | `rq2_cells.csv`, `results/extra_res/profile.csv`, `results/extra_res/latency_A.csv` | `analysis/latmodel.py` → `latmodel.txt` |
| Section 4.3, Table 3, H3a | `results/extra_res/latency_A.csv`, `results/extra_res/profile.csv` | `analysis/rq3.py` → `rq3_cells.csv` |
| Table 1 (accuracy columns), Section 4.4, H4a–H4c, Fig. 3b | `results/sc_full/accuracy.csv`, `coco.csv`, `latency_C.csv`, `sanity.csv` | `analysis/rq4.py` → `rq4_accuracy_runs.csv`, `rq4_summary.csv` |
| H1d, Fig. 2 (Pareto fronts) | `full_res/latency.csv`, `rq4_accuracy_runs.csv` | `analysis/h1d_v2.py` (corrected rule, reported) → `h1d_fronts_v2.csv`; `h1d.py` (original rule) and `h1d_noY.py` (without single-seed models) for the sensitivity analyses |
| Section 4.5, Table 4 | `results/full_res`, `results/sliceb_3080ti`, `results/sliceb_5090` (`latency.csv`, `env.jsonl`) | `analysis/sliceb_an.py` → `sliceb_pairs.csv`, `sliceb_medians.csv` |
| Online Resource 4 (BF16, CPU, YOLO26-s; INT8 layer inspection) | `results/appx/latency.csv`, `insp_layers.csv`, `insp_summary.csv` | tables read directly |

GFLOPs and parameter counts (Table 1) come from the companion repository (`analysis/complexity.py` there).

## 2. Result files

| Folder | Layer | Main files |
|---|---|---|
| `results/full_res` | core layer, RTX 4090 (16–18 Oct 2026) | `latency.csv` (one row per repetition: configuration, input size, batch, runtime cell, build, pass, repetition, median/mean/SD/p95 latency in ms over 300 timed passes, wall-clock median, compile time), `builds.csv` (TensorRT builds), `env.jsonl` (GPU UUID, driver, clocks, CPU, host and boot identifiers, library versions per block), `cfg/` (export, quantization and re-batching checks per configuration) |
| `results/stageP` | profiling layer (Stage P of `extra.py`) as used by `rq1.py` and `rq2.py` | `profile.csv` (per cell: executed kernels, launch API calls, graph launches, device time T_dev as the union of kernel intervals, leaf-module calls) |
| `results/extra_res` | profiling layer and Slice A (19–20 Oct 2026) | `latency_A.csv` (Slice A latency with MPS fraction), `profile.csv` (Stage P and Slice A), `prof/` (per-cell Nsight summaries and per-name kernel counts) |
| `results/sc_full` | Slice C (21–22 Oct 2026) | `accuracy.csv` (mAP per seed, path and split), `coco.csv` (pycocotools AP, AP_small, ...), `reference.csv` (check against the companion study), `sanity.csv` (output differences to eager FP32), `latency_C.csv` (forward and NMS time at batch 1), `calib_list.json` (the 512 calibration images) |
| `results/sliceb_3080ti`, `results/sliceb_5090` | Slice B (23 Oct 2026) | as `full_res` |
| `results/appx` | context measurements and INT8 inspection (24 Oct 2026) | `latency.csv`, `insp_layers.csv` (precision, tactic and time of every TensorRT layer), `insp_summary.csv`, `insp/` |

Runtime cell names: `eager`, `eager_graph`, `compile_default`, `compile_ro` (reduce-overhead), `ort_iobinding`, `ort_with_copies`, `trt_fp32_graph`, `trt_fp16_graph`, `trt_fp16_nograph`, `trt_int8a_graph`, `trt_int8b_graph`; `ref_eager` is the drift reference (baseline, eager, batch 1) measured at the start, middle and end of every block. Configuration names follow the companion study (`base640`, `p2_640`, `afterA`, `Bnarrow`, `c1bn`, `Bg4`, `seg4`, `c1pos`, `Awide`, `Bplain`, `seg1`, `safew288`, `y11m`, `y11l`, `base960`, `base1024`, `base1120`, `p2_832`, `p2_960`). Log files of all runs are in the Zenodo archive.

## 3. Repeating the measurements (GPU)

Environment of the primary machine (recorded in `results/*/env.jsonl`): NVIDIA RTX 4090, driver 610.57, Python 3.12.11, PyTorch 2.13.0 (CUDA 13.2), TensorRT 11.4.0.106, ONNX Runtime GPU 1.31.0, ONNX 1.23.2, Ultralytics 8.4.143, Nsight Systems; TensorRT Model Optimizer 0.47.0 in a separate virtual environment, because it cannot be installed next to this PyTorch version. See `requirements-gpu.txt`.

1. Install Ultralytics 8.4.143 and register the custom modules with `tools/patch_ultralytics.py` from the [companion repository](https://github.com/langyuan8080-ai/capacity-before-design).
2. Download the trained weights from [Zenodo](https://doi.org/10.5281/zenodo.23631046) (`weights_best_68.tar`, 68 runs listed in `tools/weights_runs.txt`, checksums in `SHA256SUMS_weights.txt`) and unpack them so that every run is at `/workspace/runs/detect/<run>/weights/best.pt`; prepare VisDrone2019-DET under `/workspace/data` as described in the companion repository.
3. Set `PV_PY` (Python of the measurement environment) and `QV_PY` (Python of the Model Optimizer environment) if they differ from the defaults at the top of `measure/core.py`.
4. Run each layer in the background and follow its progress, for example

```bash
nohup python measure/core.py run --out /workspace/paper2_core/full > core.out 2>&1 &
python measure/core.py watch /workspace/paper2_core/full      # progress bar; Ctrl+C leaves the job running
```

and likewise `extra.py`, `slicec.py`, `sliceb.py` and `appx.py` (each script documents its options in its header; `extra.py` and `appx.py` reuse the calibrated INT8 graphs of the core run through `FULL_DIR`). Latency measurements need an otherwise idle GPU; absolute values depend on the GPU, the host CPU and the library versions.

## License

The code is released under the GNU Affero General Public License v3.0 (`LICENSE`), because it uses Ultralytics (AGPL-3.0). The measurement data in `results/` are released under CC BY 4.0. VisDrone2019-DET is not redistributed.

## Citation

See `CITATION.cff`; the reference will be updated on publication.

Contact: Jin-whan Kim (kjw@ysu.ac.kr), LinFang Lang (langyuan8080@gmail.com), Youngsan University.
