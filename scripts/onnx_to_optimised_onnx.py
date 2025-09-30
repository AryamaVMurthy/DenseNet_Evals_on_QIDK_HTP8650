#!/usr/bin/env python3
import argparse, os, sys, qai_hub as hub

def main():
    ap = argparse.ArgumentParser(description="ONNX -> optimized ONNX via AI Hub")
    ap.add_argument("onnx", help="Path to source .onnx")
    ap.add_argument("--device", default="", help='Exact name or filters; e.g. "Samsung Galaxy S24 (Family)".')
    args = ap.parse_args()

    device = hub.Device(attributes="chipset:qualcomm-snapdragon-8gen3") if not args.device else hub.Device(args.device)

    job = hub.submit_compile_job(
        model=args.onnx,
        device=device,
        options="--target_runtime onnx",
    )
    print(f"SUBMITTED: compile (optimized ONNX)\nID: {job.id}\nURL: https://app.aihub.qualcomm.com/jobs/{job.id}")

if __name__ == "__main__":
    sys.exit(main())
