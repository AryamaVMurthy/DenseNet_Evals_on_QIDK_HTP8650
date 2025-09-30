#!/usr/bin/env python3
import argparse, sys, qai_hub as hub

def main():
    ap = argparse.ArgumentParser(description="Profile a QNN DLC on device via AI Hub")
    ap.add_argument("dlc", help="Path to .dlc")
    ap.add_argument("--device", default="")
    ap.add_argument("--compute-unit", default="", help='e.g. "npu", "gpu", "cpu"')
    args = ap.parse_args()

    device = hub.Device(attributes="chipset:qualcomm-snapdragon-8gen3") if not args.device else hub.Device(args.device)
    opts = f"--compute_unit {args.compute_unit}" if args.compute_unit else ""

    job = hub.submit_profile_job(
        model=args.dlc,
        device=device,
        options=opts
    )
    print(f"SUBMITTED: profile (DLC)\nID: {job.id}\nURL: https://app.aihub.qualcomm.com/jobs/{job.id}")

if __name__ == "__main__":
    sys.exit(main())
