#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


def log(msg: str):
    print(f"[classify_labels] {msg}")


def run_qnn(qnn_sdk_root: Path, context: Path, binary: Path, images_dir: Path, out_dir: Path, max_images: int) -> None:
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
        str(out_dir),
        "--max_images",
        str(max_images),
    ]
    log("Running qnn pipeline …")
    subprocess.run(cmd, check=True)


def load_imagenet_index(path: Path | None) -> Dict[int, Tuple[str, str]]:
    if path and path.exists():
        j = json.loads(path.read_text())
        return {int(k): (v[0], v[1]) for k, v in j.items()}
    # Try a few mirrors
    import urllib.request

    urls = [
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
    ]
    last_err = None
    for url in urls:
        try:
            with urllib.request.urlopen(url) as r:
                j = json.load(r)
            return {int(k): (v[0], v[1]) for k, v in j.items()}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load ImageNet class index mapping. Last error: {last_err}")


def dequantize_logits(context_json: Path, raw: bytes, expected_classes: int) -> np.ndarray:
    ctx = json.loads(context_json.read_text())
    out = ctx["info"]["graphs"][0]["info"]["graphOutputs"][0]["info"]
    qp = out.get("quantizeParams", {}).get("scaleOffset", {})
    scale = qp.get("scale")
    offset = qp.get("offset")
    if len(raw) == expected_classes:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        if scale is not None and offset is not None:
            arr = (arr - offset) * scale
        return arr
    elif len(raw) == expected_classes * 4:
        return np.frombuffer(raw, dtype=np.float32)
    else:
        raise RuntimeError(f"Unexpected logits size {len(raw)} bytes (expected {expected_classes} or {expected_classes*4})")


def classify_outputs(
    outputs_dir: Path,
    context_json: Path,
    idx_map: Dict[int, Tuple[str, str]],
    topk: int,
    allow_wnids: set[str] | None = None,
) -> List[Dict[str, str]]:
    # Determine num classes from context
    ctx = json.loads(context_json.read_text())
    out = ctx["info"]["graphs"][0]["info"]["graphOutputs"][0]["info"]
    num_classes = int(out["dimensions"][1]) if out["rank"] == 2 else int(np.prod(out["dimensions"]))

    # Iterate results
    results = []
    result_dirs = sorted([p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("Result_")], key=lambda p: int(p.name.split("_")[-1]))
    for rd in result_dirs:
        of = rd / "class_logits.raw"
        if not of.exists():
            continue
        logits = dequantize_logits(context_json, of.read_bytes(), num_classes)
        # If an allowlist is provided, mask out disallowed classes
        if allow_wnids is not None:
            allow_idx = {i for i, (w, _) in idx_map.items() if w in allow_wnids}
            if allow_idx:
                mask = np.ones_like(logits, dtype=bool)
                mask[list(allow_idx)] = False
                logits[mask] = -np.inf
        order = np.argsort(-logits)
        k = max(1, topk)
        top_idx = order[:k]
        top = [(int(i), *idx_map.get(int(i), (str(int(i)), "?")), float(logits[int(i)])) for i in top_idx]
        # Console print
        ridx = rd.name.split("_")[-1]
        print(f"{rd.name}:")
        for rank, (i, wnid, name, score) in enumerate(top, 1):
            print(f"  Top{rank}: {i:4d}  {wnid:>10}  {name:<25}  score={score:.3f}")
        results.append({
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
    return results


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    import csv

    if not rows:
        log("No results to write.")
        return
    keys = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"Saved labels CSV: {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Run QNN classification and print ImageNet labels (Top-K)")
    ap.add_argument("--qnn_sdk_root", type=Path, default=os.environ.get("QNN_SDK_ROOT", ""), help="QNN SDK root path")
    ap.add_argument("--context", type=Path, default=Path("densenet-w8a8.context.json"))
    ap.add_argument("--binary", type=Path, default=Path("densenet-w8a8.bin"))
    ap.add_argument("--image", type=Path, default=None, help="Path to a single image")
    ap.add_argument("--images_dir", type=Path, default=None, help="Directory with images (used if --image not set)")
    ap.add_argument("--out", type=Path, default=Path("oneshot"))
    ap.add_argument("--max_images", type=int, default=1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--imagenet_index_json", type=Path, default=None)
    # Label restriction options
    ap.add_argument("--allow_wnids_file", type=Path, default=None, help="File with one WNID per line to restrict labels")
    ap.add_argument("--allow_tiny_imagenet_root", type=Path, default=None, help="If set, restrict labels to wnids.txt in this Tiny-ImageNet root")
    ap.add_argument("--skip_run", action="store_true", help="Do not run qnn; only parse outputs in output_android/output")
    args = ap.parse_args()

    if not args.skip_run:
        if not args.qnn_sdk_root:
            raise RuntimeError("--qnn_sdk_root not provided and QNN_SDK_ROOT not set")
        qnn_sdk_root = args.qnn_sdk_root.resolve()
        # Prepare images_dir
        if args.image:
            # Copy single image into a temp staging dir to ensure it's the only one processed
            staging = Path(tempfile.mkdtemp(prefix="classify_single_"))
            dst = staging / args.image.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.image, dst)
            images_dir = staging
            max_images = 1
        elif args.images_dir:
            images_dir = args.images_dir
            max_images = args.max_images
        else:
            raise RuntimeError("Provide either --image or --images_dir")

        # Run QNN pipeline (this will also generate profiling and pull outputs to output_android/)
        run_qnn(qnn_sdk_root, args.context, args.binary, images_dir, args.out, max_images)

    # Load mapping
    idx_map = load_imagenet_index(args.imagenet_index_json)
    allow_wnids: set[str] | None = None
    if args.allow_wnids_file and args.allow_wnids_file.exists():
        allow_wnids = {ln.strip() for ln in args.allow_wnids_file.read_text().splitlines() if ln.strip()}
    if args.allow_tiny_imagenet_root:
        wnids_file = args.allow_tiny_imagenet_root / "wnids.txt"
        if wnids_file.exists():
            tset = {ln.strip() for ln in wnids_file.read_text().splitlines() if ln.strip()}
            allow_wnids = tset if allow_wnids is None else (allow_wnids & tset)

    # Parse outputs and print labels
    outputs_dir = Path("output_android/output")
    rows = classify_outputs(outputs_dir, args.context, idx_map, args.topk, allow_wnids=allow_wnids)
    write_csv(rows, Path("output_android/labels.csv"))


if __name__ == "__main__":
    main()
