#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

import numpy as np


def log(msg: str):
    print(f"[run_end2end_eval] {msg}")


def run_pipeline(qnn_sdk_root: Path, context: Path, binary: Path, images_dir: Path, out_name: str, max_images: int, backend: str | None = None) -> None:
    cmd = [
        sys.executable,
        "run_qidk_densenet.py",
        "--qnn_sdk_root",
        str(qnn_sdk_root),
        "--context",
        str(context),
        "--binary",
        str(binary),
        "--images_dir",
        str(images_dir),
        "--out",
        out_name,
        "--max_images",
        str(max_images),
    ]
    if backend:
        cmd += ["--backend", backend]
    log("Running device pipeline (preprocess -> push -> run -> pull -> profile)")
    subprocess.run(cmd, check=True)


def load_imagenet_index(mapping_path: Path | None) -> Dict[int, Tuple[str, str]]:
    if mapping_path and mapping_path.exists():
        j = json.loads(mapping_path.read_text())
        return {int(k): (v[0], v[1]) for k, v in j.items()}
    import urllib.request

    for url in (
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
    ):
        try:
            with urllib.request.urlopen(url) as r:
                j = json.load(r)
            return {int(k): (v[0], v[1]) for k, v in j.items()}
        except Exception:
            continue
    raise RuntimeError("Unable to load ImageNet class index mapping; set --imagenet_index_json")


def dequantize_logits(context_json: Path, raw: bytes, expected_classes: int) -> np.ndarray:
    ctx = json.loads(context_json.read_text())
    out = ctx["info"]["graphs"][0]["info"]["graphOutputs"][0]["info"]
    qp = out.get("quantizeParams", {}).get("scaleOffset", {})
    scale = qp.get("scale")
    offset = qp.get("offset")
    out_dtype = str(out.get("dataType", ""))

    # Handle common output formats: U8, U16/S16 fixed point, or FLOAT32
    if len(raw) == expected_classes:
        # Likely U8
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        if scale is not None and offset is not None:
            arr = (arr - offset) * scale
        return arr
    elif len(raw) == expected_classes * 2:
        # 16-bit output (UFIXED16 or SFIXED16)
        if "S" in out_dtype or "INT16" in out_dtype:
            dt = np.int16
        else:
            dt = np.uint16
        arr = np.frombuffer(raw, dtype=dt).astype(np.float32)
        if scale is not None and offset is not None:
            arr = (arr - offset) * scale
        return arr
    elif len(raw) == expected_classes * 4:
        # FLOAT32 logits
        return np.frombuffer(raw, dtype=np.float32)
    else:
        raise RuntimeError(
            f"Unexpected logits size {len(raw)} bytes for dtype {out_dtype} "
            f"(expected {expected_classes}, {expected_classes*2}, or {expected_classes*4})"
        )


def collect_results(outputs_dir: Path, context_json: Path, idx_map: Dict[int, Tuple[str, str]], topk: int) -> List[Dict[str, str]]:
    ctx = json.loads(context_json.read_text())
    out = ctx["info"]["graphs"][0]["info"]["graphOutputs"][0]["info"]
    num_classes = int(out["dimensions"][1]) if out["rank"] == 2 else int(np.prod(out["dimensions"]))

    rows: List[Dict[str, str]] = []
    rdirs = sorted([p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("Result_")], key=lambda p: int(p.name.split("_")[-1]))
    for rd in rdirs:
        of = rd / "class_logits.raw"
        if not of.exists():
            continue
        logits = dequantize_logits(context_json, of.read_bytes(), num_classes)
        order = np.argsort(-logits)
        k = max(1, topk)
        top_idx = order[:k]
        top = [(int(i), *idx_map.get(int(i), (str(int(i)), "?")), float(logits[int(i)])) for i in top_idx]
        rows.append({
            "result": rd.name,
            "top1_idx": str(top[0][0]),
            "top1_wnid": top[0][1],
            "top1_name": top[0][2],
            "top1_score": f"{top[0][3]:.6f}",
            "topk_idx": ",".join(str(t[0]) for t in top),
            "topk_wnid": ",".join(t[1] for t in top),
            "topk_name": ",".join(t[2] for t in top),
            "topk_score": ",".join(f"{t[3]:.6f}" for t in top),
        })
    return rows


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def tiny_imagenet_truth(root: Path, manifest: Path) -> List[Tuple[Path, str]]:
    # manifest: index,raw,src
    mapping = {}
    for line in (root / "val" / "val_annotations.txt").read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    pairs: List[Tuple[Path, str]] = []
    with manifest.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            src = Path(r["src"]).name
            wnid = mapping.get(src)
            if wnid is None:
                wnid = "unknown"
            pairs.append((Path(r["src"]), wnid))
    return pairs


def compute_accuracy(labels_csv: Path, truth_pairs: List[Tuple[Path, str]], topk: int) -> Dict[str, float]:
    # Load predictions
    preds = {}
    with labels_csv.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            preds[r["result"]] = r
    # Results are in order Result_0, Result_1 ... matching manifest index
    ok1 = 0
    okk = 0
    n = 0
    for i, (src, true_wnid) in enumerate(truth_pairs):
        rid = f"Result_{i}"
        pr = preds.get(rid)
        if not pr:
            continue
        n += 1
        if pr["top1_wnid"] == true_wnid:
            ok1 += 1
        if topk > 1:
            topk_wnids = pr["topk_wnid"].split(",")
            if true_wnid in topk_wnids:
                okk += 1
    return {
        "count": n,
        "top1": ok1 / n if n else 0.0,
        "topk": okk / n if n and topk > 1 else 0.0,
    }


def parse_profile_text(profile_path: Path) -> Dict[str, float]:
    text = profile_path.read_text(errors="ignore")
    acc_times = [int(x) for x in re.findall(r"Accelerator \(execute\) time\s*:\s*(\d+)\s*us", text)]
    qnn_times = [int(x) for x in re.findall(r"QNN \(execute\) time\s*:\s*(\d+)\s*us", text)]
    # Fallback: try scan all numbers after "Execute Stat" blocks if needed
    prof = {}
    if acc_times:
        prof.update({
            "acc_execute_us_avg": float(mean(acc_times)),
            "acc_execute_us_p50": float(median(acc_times)),
            "acc_execute_us_min": float(min(acc_times)),
            "acc_execute_us_max": float(max(acc_times)),
        })
    if qnn_times:
        try:
            p90 = float(np.percentile(qnn_times, 90))
            p95 = float(np.percentile(qnn_times, 95))
            p99 = float(np.percentile(qnn_times, 99))
        except Exception:
            p90 = p95 = p99 = 0.0
        prof.update({
            "qnn_execute_us_avg": float(mean(qnn_times)),
            "qnn_execute_us_p50": float(median(qnn_times)),
            "qnn_execute_us_p90": p90,
            "qnn_execute_us_p95": p95,
            "qnn_execute_us_p99": p99,
            "qnn_execute_us_min": float(min(qnn_times)),
            "qnn_execute_us_max": float(max(qnn_times)),
            "ips": 1e6 / float(mean(qnn_times)) if mean(qnn_times) > 0 else 0.0,
        })
    return prof


def main():
    ap = argparse.ArgumentParser(description="End-to-end run + accuracy + profiling summary")
    ap.add_argument("--qnn_sdk_root", type=Path, default=os.environ.get("QNN_SDK_ROOT", ""))
    ap.add_argument("--context", type=Path, default=Path("densenet-w8a8.context.json"))
    ap.add_argument("--binary", type=Path, default=Path("densenet-w8a8.bin"))
    ap.add_argument("--images_dir", type=Path, required=True)
    ap.add_argument("--dataset", choices=["tiny-imagenet", "imagenet-v2", "custom"], default="tiny-imagenet")
    ap.add_argument("--labels_csv", type=Path, help="For dataset=custom, CSV mapping with columns: filename,wnid")
    ap.add_argument("--imagenet_index_json", type=Path, default=None)
    ap.add_argument("--max_images", type=int, default=50)
    ap.add_argument("--topk_max", type=int, default=10, help="Collect top-K up to this value (enables top-1/5/10 accuracy)")
    ap.add_argument("--out", default="evalrun")
    # Optional backend override forwarded to run_qidk_densenet.py
    ap.add_argument("--backend", choices=["auto", "htp", "cpu"], default=None)
    args = ap.parse_args()

    if not args.qnn_sdk_root:
        raise RuntimeError("--qnn_sdk_root not provided and QNN_SDK_ROOT not set")

    # 1) Run pipeline on device
    # Auto: float I/O -> CPU, others -> HTP. Let run_qidk_densenet.py decide by default.
    run_pipeline(
        args.qnn_sdk_root.resolve(),
        args.context,
        args.binary,
        args.images_dir,
        args.out,
        args.max_images,
        backend=args.backend,
    )

    # 2) Label predictions
    idx_map = load_imagenet_index(args.imagenet_index_json)
    labels_rows = collect_results(Path("output_android/output"), args.context, idx_map, args.topk_max)
    labels_csv = Path("output_android/labels.csv")
    write_csv(labels_rows, labels_csv)
    log(f"Wrote labels: {labels_csv}")

    # 3) Accuracy (dataset-specific)
    acc = None
    if args.dataset == "tiny-imagenet":
        manifest = Path(args.out) / "input_manifest.csv"
        truth_pairs = tiny_imagenet_truth(Path("tiny-imagenet-200"), manifest)
        acc1 = compute_accuracy(labels_csv, truth_pairs, 1)
        acc5 = compute_accuracy(labels_csv, truth_pairs, 5)
        acc10 = compute_accuracy(labels_csv, truth_pairs, 10)
        log(f"Top-1 acc: {acc1['top1']*100:.2f}%  Top-5 acc: {acc5['topk']*100:.2f}%  Top-10 acc: {acc10['topk']*100:.2f}%  (N={acc1['count']})")
        acc = {"top1": acc1['top1'], "top5": acc5['topk'], "top10": acc10['topk'], "count": acc1['count']}
    elif args.dataset == "imagenet-v2":
        # Derive ground truth from parent folder name (WNID) for each image in manifest
        manifest = Path(args.out) / "input_manifest.csv"
        truth_pairs: List[Tuple[Path, str]] = []
        with manifest.open() as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                src = Path(r["src"])  # original image path
                wnid = src.parent.name  # folder name is WNID in ImageNetV2 format-val
                truth_pairs.append((src, wnid))
        acc1 = compute_accuracy(labels_csv, truth_pairs, 1)
        acc5 = compute_accuracy(labels_csv, truth_pairs, 5)
        acc10 = compute_accuracy(labels_csv, truth_pairs, 10)
        log(f"Top-1 acc: {acc1['top1']*100:.2f}%  Top-5 acc: {acc5['topk']*100:.2f}%  Top-10 acc: {acc10['topk']*100:.2f}%  (N={acc1['count']})")
        acc = {"top1": acc1['top1'], "top5": acc5['topk'], "top10": acc10['topk'], "count": acc1['count']}
    elif args.dataset == "custom" and args.labels_csv:
        # TODO: implement generic CSV label matching
        log("Custom dataset accuracy not implemented; labels written only.")

    # 4) Profiling summary
    profile_txt = Path("output_android/profile_htp.csv")  # textual file from qnn-profile-viewer
    prof = parse_profile_text(profile_txt) if profile_txt.exists() else {}
    if prof:
        log("Profiling summary (per-inference):")
        for k, v in prof.items():
            if k.endswith("ips"):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v:.2f}")

    # 5) Summary JSON
    summary = {
        "accuracy": acc or {},
        "profiling": prof,
        "labels_csv": str(labels_csv),
        "profile_file": str(profile_txt),
    }
    (Path("output_android") / "summary.json").write_text(json.dumps(summary, indent=2))
    log("Done.")


if __name__ == "__main__":
    main()
