#!/usr/bin/env python3
"""
Quantize MobileNetV3 ONNX to W8A8 and compile to QNN via Qualcomm AI Hub only.

Flow
- Build a small calibration dataset from images (float32, correct shape/layout)
- Upload ONNX and calibration dataset to AI Hub
- Quantize to INT8 weights + INT8 activations (QDQ ONNX)
- Compile the quantized model for the selected device (QNN target)
- Optionally generate a QNN Context Binary (link job) if desired
- Download compiled artifacts locally

Usage
  python3 aihub_quantize_compile_mobilenetv3.py \
    --onnx model.onnx \
    --calib_dir tiny-imagenet-200/val/images \
    --device "Samsung Galaxy S24 (Family)" \
    --samples 300 --size 224 --normalize raw \
    --out_dir outputs/aihub/mobilenetv3

Notes
- Requires `qai_hub` SDK installed and configured with an API token.
- Pillow and (optionally) onnx Python packages. If onnx is not installed, the
  script defaults to input (1,3,224,224) named "input".
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import qai_hub as hub


def try_detect_onnx_spec(onnx_path: Path) -> Tuple[str, Tuple[int, ...]]:
    try:
        import onnx  # type: ignore
        m = onnx.load(str(onnx_path))
        g = m.graph
        if not g.input:
            return ("input", (1, 3, 224, 224))
        # Prefer common name 'image_tensor' if present
        inputs = list(g.input)
        inp = None
        for i in inputs:
            if (i.name or '').lower() == 'image_tensor':
                inp = i
                break
        if inp is None:
            inp = inputs[0]
        name = inp.name or "input"
        dims = []
        shp = inp.type.tensor_type.shape.dim
        for i, d in enumerate(shp):
            v = d.dim_value if d.dim_value > 0 else (1 if i == 0 else 224)
            dims.append(int(v))
        if len(dims) == 3:
            dims = [1] + dims
        if len(dims) < 4:
            dims = [1, 3, 224, 224]
        return name, tuple(dims)
    except Exception:
        return ("input", (1, 3, 224, 224))


def load_images_as_entries(img_dir: Path, samples: int, size: int, input_name: str, dims: Tuple[int, ...], normalize: str) -> dict:
    try:
        from PIL import Image
    except Exception as e:
        raise SystemExit("Missing dependency 'Pillow'. Install it and retry.")

    # Layout detection
    if len(dims) < 4:
        dims = (1, 3, size, size)
    if dims[1] == 3:
        layout = 'NCHW'
        H, W, C = dims[2], dims[3], dims[1]
    else:
        layout = 'NHWC'
        H, W, C = dims[1], dims[2], dims[3]

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG'}
    paths = [p for p in sorted(img_dir.iterdir()) if p.suffix in exts]
    if not paths:
        raise SystemExit(f"No images found in {img_dir}")
    paths = paths[:samples]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    entries: List[np.ndarray] = []
    for p in paths:
        im = Image.open(p).convert('RGB').resize((W, H), Image.BILINEAR)
        a = np.asarray(im).astype(np.float32) / 255.0  # H,W,C in [0,1]
        if normalize == 'imagenet':
            a = (a - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        if layout == 'NCHW':
            a = a.transpose(2, 0, 1)  # C,H,W
        a = np.expand_dims(a, axis=0)  # 1,C,H,W or 1,H,W,C
        entries.append(a.astype(np.float32))

    return {input_name: entries}


def main():
    ap = argparse.ArgumentParser(description="Quantize + compile MobileNetV3 ONNX on AI Hub")
    ap.add_argument('--onnx', type=Path, default=Path('model.onnx'), help='Path to MobileNetV3 ONNX file')
    ap.add_argument('--calib_dir', type=Path, default=Path('tiny-imagenet-200/val/images'), help='Calibration images dir')
    ap.add_argument('--device', type=str, default='Samsung Galaxy S24 (Family)')
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--normalize', choices=['raw', 'imagenet'], default='raw')
    ap.add_argument('--input_name', type=str, default=None)
    ap.add_argument('--input_shape', type=str, default=None, help='Override, e.g., 1,3,224,224')
    ap.add_argument('--out_dir', type=Path, default=Path('outputs/aihub/mobilenetv3'))
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    if not args.calib_dir.exists():
        raise SystemExit(f"Calibration images dir not found: {args.calib_dir}")

    name, dims = try_detect_onnx_spec(args.onnx)
    if args.input_name:
        name = args.input_name
    if args.input_shape:
        try:
            dims = tuple(int(x) for x in args.input_shape.split(','))
        except Exception:
            raise SystemExit("--input_shape must be comma-separated integers, e.g., 1,3,224,224")

    print(f"Input spec: name={name}, shape={dims}")

    # Build calibration dataset entries and upload
    data_entries = load_images_as_entries(args.calib_dir, args.samples, args.size, name, dims, args.normalize)
    dataset = hub.upload_dataset(data_entries, name='mobilenetv3-calibration')
    print(f"Uploaded calibration dataset id={dataset.dataset_id}")

    # Quantize to INT8/INT8 (QDQ ONNX)
    qjob = hub.submit_quantize_job(
        model=str(args.onnx),
        calibration_data=dataset,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name='mobilenetv3-w8a8',
    )
    print(f"Quantize job: {qjob.url}")
    qjob.wait()
    q_model = qjob.get_target_model()
    if q_model is None:
        raise SystemExit(f"Quantize job failed: {getattr(qjob, 'failure_reason', 'unknown')}")
    print("Quantized ONNX ready")

    # Compile for device (QNN)
    device = hub.Device(args.device)
    cjob = hub.submit_compile_job(
        model=q_model,
        device=device,
        name='mobilenetv3-w8a8-qnn',
        input_specs={name: tuple(dims)},
    )
    print(f"Compile job: {cjob.url}")
    cjob.wait()
    if not cjob.success:
        raise SystemExit(f"Compile job failed: {cjob.failure_reason}")

    # Download compiled artifacts
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cjob.download_results(str(out_dir / 'compiled'))
    print(f"Downloaded compiled artifacts -> {out_dir/'compiled'}")

    # Try to generate a context binary (optional)
    try:
        ljob = hub.submit_link_job(models=cjob.get_target_model(), device=device, name='mobilenetv3-context')
        print(f"Link job: {ljob.url}")
        ljob.wait()
        if ljob.success:
            ljob.download_results(str(out_dir / 'context'))
            print(f"Downloaded context binary -> {out_dir/'context'}")
        else:
            print(f"Link job failed: {ljob.failure_reason}")
    except Exception as e:
        print(f"Skipping link job (context binary): {e}")

    print("Done.")


if __name__ == '__main__':
    main()
