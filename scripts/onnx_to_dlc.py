#!/usr/bin/env python3
import argparse, sys, qai_hub as hub

def main():
    ap = argparse.ArgumentParser(description="ONNX -> QNN DLC via AI Hub")
    ap.add_argument("onnx", help="Path to source .onnx")
    ap.add_argument("--device", default="")
    ap.add_argument("--compile-options", default="--target_runtime qnn_dlc", help="Extra compile options")
    args = ap.parse_args()

    device = hub.Device(attributes="chipset:qualcomm-snapdragon-8gen3") if not args.device else hub.Device(args.device)

    job = hub.submit_compile_job(
        model=args.onnx,
        device=device,
        options=args.compile_options,
    )
    print(f"SUBMITTED: compile (ONNX->DLC)\nID: {job.id}\nURL: https://app.aihub.qualcomm.com/jobs/{job.id}")

if __name__ == "__main__":
    sys.exit(main())
