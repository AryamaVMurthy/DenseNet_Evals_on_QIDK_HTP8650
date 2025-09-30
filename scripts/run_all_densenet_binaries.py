#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

def log(msg: str):
    print(f"[run_all_densenet_binaries] {msg}")

def ensure_context(qnn_sdk_root: Path, context_bin: Path, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(qnn_sdk_root / "bin/x86_64-linux-clang/qnn-context-binary-utility"),
        "--context_binary", str(context_bin),
        "--json_file", str(out_json),
    ]
    subprocess.run(cmd, check=True)

def read_context_meta(ctx_json: Path) -> Tuple[int, str, str]:
    """Return (dspArch, input_dtype, output_dtype) from a context JSON."""
    j = json.loads(ctx_json.read_text())
    info = j["info"]
    dspArch = int(info.get("contextMetadata", {}).get("info", {}).get("dspArch", -1))
    g = info["graphs"][0]["info"]
    in_dtype = str(g["graphInputs"][0]["info"]["dataType"])
    out_dtype = str(g["graphOutputs"][0]["info"]["dataType"])
    return dspArch, in_dtype, out_dtype

def run_single(qnn_sdk_root: Path, binary: Path, context_json: Path, images_dir: Path,
               dataset: str, max_images: int, topk_max: int, work_out: str, backend: str | None = None) -> Dict:
    cmd = [
        "python", "run_end2end_eval.py",
        "--qnn_sdk_root", str(qnn_sdk_root),
        "--context", str(context_json),
        "--binary", str(binary),
        "--images_dir", str(images_dir),
        "--dataset", dataset,
        "--max_images", str(max_images),
        "--topk_max", str(topk_max),
        "--out", work_out,
    ]
    if backend:
        cmd += ["--backend", backend]
    log(f"Running eval for {binary.name}")
    subprocess.run(cmd, check=True)
    summary_path = Path("output_android") / "summary.json"
    return json.loads(summary_path.read_text())

def main():
    ap = argparse.ArgumentParser(description="Evaluate all DenseNet QNN context binaries and aggregate metrics.")
    ap.add_argument("--qnn_sdk_root", type=Path, default=os.environ.get("QNN_SDK_ROOT", ""), help="QNN SDK root")
    ap.add_argument("--binaries_dir", type=Path, default=Path("densenet_binaries"), help="Folder with *.bin binaries")
    ap.add_argument("--images_dir", type=Path, required=True, help="Image root (e.g., imagenetv2-sample)")
    ap.add_argument("--dataset", choices=["imagenet-v2", "tiny-imagenet", "custom"], default="imagenet-v2")
    ap.add_argument("--max_images", type=int, default=1000)
    ap.add_argument("--topk_max", type=int, default=5)
    ap.add_argument("--results_dir", type=Path, default=Path("multi_eval_results"))
    args = ap.parse_args()

    if not args.qnn_sdk_root:
        raise RuntimeError("--qnn_sdk_root not provided and QNN_SDK_ROOT not set")
    qnn_sdk_root = args.qnn_sdk_root.resolve()

    bins: List[Path] = sorted(args.binaries_dir.glob("*.bin"))
    if not bins:
        raise RuntimeError(f"No .bin files found in {args.binaries_dir}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    table_rows: List[Dict] = []

    for b in bins:
        base = b.stem
        ctx_json = b.with_suffix(".context.json")
        try:
            if not ctx_json.exists():
                ensure_context(qnn_sdk_root, b, ctx_json)
            dspArch, in_dtype, out_dtype = read_context_meta(ctx_json)
        except Exception as e:
            log(f"Skipping {b.name}: cannot read/generate context JSON ({e})")
            continue

        # Skip FLOAT I/O models — only run INT (quantized) binaries end to end
        if "FLOAT" in str(in_dtype) or "FLOAT" in str(out_dtype) or "float" in b.name.lower():
            log(f"Skipping FLOAT model: {b.name} (in={in_dtype}, out={out_dtype})")
            continue

        work_out = f"eval_{base}"
        try:
            # For FLOAT I/O contexts, force CPU backend to avoid HTP crashes
            backend = "cpu" if str(in_dtype).startswith("QNN_DATATYPE_FLOAT") else None
            summary = run_single(qnn_sdk_root, b, ctx_json, args.images_dir, args.dataset, args.max_images, args.topk_max, work_out, backend=backend)
        except subprocess.CalledProcessError as e:
            log(f"Run failed for {b.name}: {e}")
            continue

        out_dir = args.results_dir / base
        out_dir.mkdir(parents=True, exist_ok=True)
        # snapshot artifacts
        for fname in ["summary.json", "labels.csv", "profile_htp.csv", "report.md"]:
            src = Path("output_android") / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        with (out_dir / "run_info.txt").open("w") as f:
            f.write(f"binary={b}\ncontext={ctx_json}\ndspArch={dspArch}\nin_dtype={in_dtype}\nout_dtype={out_dtype}\nwork_out={work_out}\n")

        acc = summary.get("accuracy", {})
        prof = summary.get("profiling", {})
        row = {
            "binary": b.name,
            "dspArch": dspArch,
            "in_dtype": in_dtype,
            "out_dtype": out_dtype,
            "count": acc.get("count", 0),
            "top1": acc.get("top1", 0.0),
            "top5": acc.get("top5", acc.get("topk", 0.0)),
            "qnn_avg_us": prof.get("qnn_execute_us_avg", 0.0),
            "qnn_p50_us": prof.get("qnn_execute_us_p50", 0.0),
            "ips": prof.get("ips", 0.0),
        }
        table_rows.append(row)
        log(f"Done: {b.name}  N={row['count']}  top1={row['top1']*100:.2f}%  top5={row['top5']*100:.2f}%  ips={row['ips']:.2f}")

    agg_csv = args.results_dir / "results.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["binary", "dspArch", "in_dtype", "out_dtype", "count", "top1", "top5", "qnn_avg_us", "qnn_p50_us", "ips"])
        writer.writeheader()
        writer.writerows(table_rows)
    log(f"Aggregate results written to {agg_csv}")

if __name__ == "__main__":
    main()
