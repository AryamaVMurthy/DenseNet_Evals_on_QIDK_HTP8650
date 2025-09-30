#!/usr/bin/env python3
import argparse, os, sys, numpy as np
from PIL import Image
import qai_hub as hub

def parse_mode(mode: str):
    mode = mode.lower()
    if mode not in {"w8a8","w8a16","w4a8","w4a16"}:
        raise ValueError("mode must be one of w8a8, w8a16, w4a8, w4a16")
    w = hub.QuantizeDtype.INT8 if "w8" in mode else hub.QuantizeDtype.INT4
    a = hub.QuantizeDtype.INT16 if "a16" in mode else hub.QuantizeDtype.INT8
    return w, a

def load_dir_as_nchw_batches(img_dir, shape):
    n, c, h, w = shape
    mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    std  = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    batch = []
    for name in sorted(os.listdir(img_dir))[:100]:
        p = os.path.join(img_dir, name)
        try:
            im = Image.open(p).convert("RGB").resize((w, h))
        except Exception:
            continue
        arr = np.array(im).astype(np.float32)/255.0
        arr = np.transpose(arr, (2,0,1))  # HWC->CHW
        arr = ((arr - mean)/std).astype(np.float32)
        batch.append(arr[None, ...])  # add N
    if not batch:
        raise RuntimeError(f"No images loaded from {img_dir}")
    return batch

def main():
    ap = argparse.ArgumentParser(description="Quantize ONNX with AI Hub")
    ap.add_argument("onnx", help="Path to unquantized .onnx")
    ap.add_argument("--mode", required=True, help="w8a8 | w8a16 | w4a8 | w4a16")
    ap.add_argument("--input-name", default="image_tensor", help="ONNX input name for calibration")
    ap.add_argument("--shape", default="1,3,224,224", help="N,C,H,W for calibration samples")
    ap.add_argument("--calib-dir", default="", help="Directory of images for calibration (optional)")
    ap.add_argument("--random", action="store_true", help="Use a single random sample instead of images")
    args = ap.parse_args()

    n,c,h,w = map(int, args.shape.split(","))
    w_dtype, a_dtype = parse_mode(args.mode)

    if args.calib_dir:
        samples = load_dir_as_nchw_batches(args.calib_dir, (n,c,h,w))
    else:
        # quick benchmarking path: 1 random sample (accuracy will be poor)
        rnd = np.random.randn(n, c, h, w).astype(np.float32)
        samples = [rnd]

    calibration_data = {args.input_name: samples}

    job = hub.submit_quantize_job(
        model=args.onnx,
        calibration_data=calibration_data,
        weights_dtype=w_dtype,
        activations_dtype=a_dtype,
    )
    print(f"SUBMITTED: quantize ({args.mode})\nID: {job.id}\nURL: https://app.aihub.qualcomm.com/jobs/{job.id}")

if __name__ == "__main__":
    sys.exit(main())
