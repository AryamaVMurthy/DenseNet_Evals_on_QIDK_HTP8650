#!/usr/bin/env python3
import argparse, sys, qai_hub as hub

def main():
    ap = argparse.ArgumentParser(description="DLC -> QNN context binary via AI Hub Link Job")
    ap.add_argument("dlc", nargs="+", help="One or more .dlc paths")
    ap.add_argument("--device", default="")
    ap.add_argument("--link-options", default="", help="Optional link options")
    args = ap.parse_args()

    # Link jobs don’t require a specific device to build the binary, but accept one for consistency
    device = hub.Device(attributes="chipset:qualcomm-snapdragon-8gen3") if not args.device else hub.Device(args.device)

    job = hub.submit_link_job(
        models=args.dlc,
        device=device,
        options=args.link_options
    )
    print(f"SUBMITTED: link (DLC->context binary)\nID: {job.id}\nURL: https://app.aihub.qualcomm.com/jobs/{job.id}")

if __name__ == "__main__":
    sys.exit(main())
