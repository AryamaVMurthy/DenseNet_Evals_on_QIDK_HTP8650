# DenseNet on QIDK (HTP) — End‑to‑End Runbook

This runbook shows how to run the provided DenseNet context binaries on a QIDK Android kit, collect accuracy and profiling, and aggregate benchmarks across multiple quantization variants.

## 0) Prerequisites

- QNN SDK is unpacked at `v2.38.0.250901/qairt/2.38.0.250901` in this repo (adjust if different).
- Android device is connected and visible to `adb devices`.
- Python environment (this repo uses the included scripts).

```bash
# From repo root
export QNN_SDK_ROOT="$PWD/v2.38.0.250901/qairt/2.38.0.250901"
# Optional: set ANDROID_SERIAL if multiple devices are connected
# export ANDROID_SERIAL=<device_id>
```

## 1) Prepare an evaluation dataset (ImageNet‑V2)

Use the helper to export a small, high‑quality ImageNet‑V2 sample arranged in WNID subfolders.

```bash
# Install once if needed
python -m pip install --upgrade pillow
python -m pip install --upgrade git+https://github.com/modestyachts/ImageNetV2_pytorch

# Export 1000 or 5000 images (choose one)
python prepare_imagenetv2_to_dir.py --variant matched-frequency --out_dir imagenetv2-sample --count 1000
# or
python prepare_imagenetv2_to_dir.py --variant matched-frequency --out_dir imagenetv2-sample --count 5000
```

You can also point to any existing folder dataset arranged as `<root>/<WNID>/*.jpg` and skip this step.

## 2) Run a single model end‑to‑end (accuracy + profiling)

The evaluator runs the full device pipeline and writes:
- `output_android/labels.csv`: per‑image Top‑K labels
- `output_android/summary.json`: Top‑1/Top‑5/Top‑10 accuracies and latency summary
- `output_android/profile_htp.csv`: HTP profiling CSV (for HTP runs)

Example — quantized model (W8A8):
```bash
python run_end2end_eval.py \
  --qnn_sdk_root "$QNN_SDK_ROOT" \
  --context densenet_binaries/densenet121-w8a8-context.context.json \
  --binary  densenet_binaries/densenet121-w8a8-context.bin \
  --images_dir imagenetv2-sample \
  --dataset imagenet-v2 \
  --max_images 1000 \
  --topk_max 5
```

Notes:
- Scripts auto‑detect context I/O type and feed the correct inputs (U8/U16/FLOAT) and select the right backend:
  - Quantized contexts → HTP backend (with correct v68/v69/v73/v75/v79 stub/skel)
  - FLOAT I/O contexts → CPU backend (to avoid CDSP crashes)

## 3) Run all DenseNet binaries and aggregate results

This runs every `.bin` in `densenet_binaries/`, computes accuracy and latency, and writes:
- `multi_eval_results/<binary_stem>/` — per‑binary artifacts (summary.json, labels.csv, profile_htp.csv, report.md)
- `multi_eval_results/results.csv` — aggregate table (binary, arch, I/O dtypes, Top‑1/Top‑5, avg/p50 latency, IPS)

```bash
python run_all_densenet_binaries.py \
  --qnn_sdk_root "$QNN_SDK_ROOT" \
  --binaries_dir densenet_binaries \
  --images_dir imagenetv2-sample \
  --dataset imagenet-v2 \
  --max_images 1000 \
  --topk_max 5
```

Print a quick accuracy summary:
```bash
python - <<'PY'
import csv
p='multi_eval_results/results.csv'
with open(p) as f:
    r=csv.DictReader(f)
    for row in r:
        n=int(row['count']); top1=float(row['top1'])*100; top5=float(row['top5'])*100
        print(f"{row['binary']} (arch v{row['dspArch']}, in={row['in_dtype']}, out={row['out_dtype']}): N={n} Top-1={top1:.2f}% Top-5={top5:.2f}% IPS={float(row['ips']):.2f}")
PY
```

## 4) (Optional) Direct profiling run and report

If you want a profiling‑focused run on a single model (with an auto‑generated markdown report), use:
```bash
python run_qidk_densenet.py \
  --qnn_sdk_root "$QNN_SDK_ROOT" \
  --context densenet_binaries/densenet121-w8a8-context.context.json \
  --binary  densenet_binaries/densenet121-w8a8-context.bin \
  --images_dir imagenetv2-sample \
  --out eval_profile \
  --max_images 200 \
  --profiling_level backend \
  --htp_prof_level linting
```
This prints a concise latency summary on the terminal and writes `output_android/report.md` with links to artifacts.

## 5) (Optional) Generate context JSONs (sanity)

Use this to inspect I/O dtypes and target arch per binary:
```bash
for f in densenet_binaries/*.bin; do \
  "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility" \
  --context_binary "$f" --json_file "${f%.bin}.context.json"; \
done
```

## Troubleshooting

- "Stuck" at adb push: large runs push many input files; try `--max_images 50` first. Ask to enable tar‑push if you want faster transfers.
- FLOAT contexts crash HTP: the scripts force CPU backend for FLOAT I/O; quantized models use HTP.
- Wrong arch errors: scripts pick the right HTP stub/skel from `dspArch` in the context JSON.
- Accuracy near zero: ensure the dataset folder is WNID‑structured and `--dataset imagenet-v2` is used (ground truth derives from parent folder).

## Artifacts

- `multi_eval_results/results.csv` — aggregate accuracy + latency for all binaries
- `output_android/summary.json` — Top‑1/Top‑5/Top‑10 (single run) + latency percentiles
- `output_android/profile_htp.csv` — per‑op HTP profiling CSV (HTP runs)
- `output_android/labels.csv` — per‑image Top‑K labels and scores
