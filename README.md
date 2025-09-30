# DenseNet Quantization Accuracy Evals

This repository benchmarks a collection of quantized DenseNet (and optional MobileNet) context binaries on a QNN-supported Android device. It covers both single-model runs and batch evaluation with accuracy and latency summaries. A helper script (`eval_densenet121_fp32.py`) is also provided for full-precision host-side baselines using PyTorch.

## Repository Layout

```
densenet-quantization-accuracy-evals/
  README.md                 # this file
  scripts/
    run_all_densenet_binaries.py
    run_end2end_eval.py
    run_qidk_densenet.py
    prepare_imagenetv2_to_dir.py
    classify_labels.py
    eval_densenet121_fp32.py
  densenet_binaries/        # quantized DenseNet context binaries (.bin + .context.json)
  mobilenet_binaries/       # optional MobileNet quantized contexts (.bin + .context.json)
  datasets/                 # place download/generated datasets here
  docs/RUNBOOK.md           # detailed runbook (optional)
```

## Requirements

1. **QNN SDK** installed locally (the scripts assume `v2.38.0.250901`). Set `QNN_SDK_ROOT` accordingly.
2. **Android device/QIDK** with USB debugging enabled and accessible via `adb`.
3. **Python 3.8+** with the following packages:
   ```bash
   python -m pip install --upgrade torch torchvision pillow
   python -m pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
   ```

## Quick Start (Quantized DenseNet)

```bash
# 0. Environment
export QNN_SDK_ROOT="/path/to/qnn-sdk"

# 1. Prepare dataset (ImageNet-V2 sample, 1000 images)
python scripts/prepare_imagenetv2_to_dir.py \
  --variant matched-frequency \
  --out_dir datasets/imagenetv2-sample \
  --count 1000

# 2. Generate/refresh context JSONs for binaries (optional but recommended)
for f in densenet_binaries/*.bin; do \
  "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility" \
    --context_binary "$f" --json_file "${f%.bin}.context.json"; \
done

# 3. Run all quantized DenseNet contexts end-to-end
python scripts/run_all_densenet_binaries.py \
  --qnn_sdk_root "$QNN_SDK_ROOT" \
  --binaries_dir densenet_binaries \
  --images_dir datasets/imagenetv2-sample \
  --dataset imagenet-v2 \
  --max_images 1000 \
  --topk_max 5

# 4. Print accuracy summary
python - <<'PY'
import csv
with open('multi_eval_results/results.csv') as f:
    for row in csv.DictReader(f):
        n = int(row['count'])
        top1 = float(row['top1'])*100
        top5 = float(row['top5'])*100
        print(f"{row['binary']} (arch v{row['dspArch']}, in={row['in_dtype']}, out={row['out_dtype']}): "
              f"N={n} Top-1={top1:.2f}% Top-5={top5:.2f}%")
PY
```

Artifacts for each run are stored under `multi_eval_results/<binary>/` (including per-image labels and HTP profiling CSVs).

## Other Useful Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_qidk_densenet.py` | Run a single context on-device, generate a report, optional Chrometrace |
| `scripts/run_end2end_eval.py`  | Runs a specific context on `--images_dir` and writes `output_android/summary.json` |
| `scripts/classify_labels.py`   | Decode logits produced by `run_qidk_densenet.py` |
| `scripts/eval_densenet121_fp32.py` | Evaluate full-precision DenseNet121 on the host with PyTorch |

## Notes on Float/DLC Models

- Quantized `.bin` contexts are executed on the HTP backend.
- FLOAT contexts (input/output dtype `QNN_DATATYPE_FLOAT_*`) are skipped by default in the batch runner. Run them separately via DLC or the FP32 script.
- DLCs (e.g., `densenet121-densenet-121-float.dlc`) can be executed on-device with `qnn-net-run --dlc_path` if needed; see `run_qidk_densenet.py` for input preparation logic.

## Accuracy on Host (FP32)

To benchmark the standard FP32 DenseNet121 model locally:

```bash
python scripts/eval_densenet121_fp32.py \
  --images_dir datasets/imagenetv2-sample \
  --batch_size 32 \
  --device cuda   # switch to cpu if no GPU is available
```

