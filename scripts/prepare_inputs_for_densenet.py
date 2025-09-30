#!/usr/bin/env python3
"""
Robust prepare_inputs.py for SNPE DLCs.

This is an improved version that robustly parses `snpe-dlc-info -i <dlc>` output
to extract the input tensor's dims, type, bitwidth, scale, and offset.

Usage (example):
 python3 prepare_inputs.py \
   --dataset_dir tiny-imagenet-200/val/images \
   --dlcs densenet121-densenet-121-w8a8.dlc \
          densenet121-densenet-121-w8a8_mixed_int16.dlc \
          densenet121-densenet-121-w8a16.dlc \
          densenet121-densenet-121-float.dlc \
   --out_dir /workspace/inputs_by_dlc \
   --num_images 200 \
   --size 224 \
   --float_mode raw --resize_filter bilinear

Notes:
 - Requires Pillow and numpy.
 - snpe-dlc-info default path is /opt/qairt/bin/x86_64-linux-clang/snpe-dlc-info (override with --snpe_dlc_info).
 - If a DLC actually lacks scale/offset (rare), you can pass --force_scale and --force_offset.
 - NOTE: Default float preprocessing is now 'raw' to avoid double-normalization
         when the DLC already includes color transforms (common in AI Hub exports).
"""
import argparse
import subprocess
import re
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import os

# ---------------------------
# Utilities to call snpe-dlc-info
# ---------------------------
def run_snpe_dlc_info(snpe_bin, dlc_path):
    cmd = [snpe_bin, "-i", dlc_path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print("ERROR: snpe-dlc-info failed:\n", e.output)
        raise
    return out

# ---------------------------
# Robust parser for snpe-dlc-info output
# ---------------------------
def extract_input_info(dlc_info_text):
    """
    Returns dict: { 'dims': [1,224,224,3], 'type': 'uFxp_8' or 'Float_32', 
                    'bitwidth': 8, 'scale': float or None, 'offset': float or None }
    """
    info = {'dims': None, 'type': None, 'bitwidth': None, 'scale': None, 'offset': None}

    # Strategy:
    # 1) Try to find a table row that contains 'image_tensor' and parse the pipe-separated columns.
    # 2) If not found, fall back to searching for "Input Name" block and the next non-empty lines.
    # 3) Try to find scale=..., offset=..., bitwidth ... anywhere near that row.

    # Normalize newlines
    text = dlc_info_text

    # Regex attempt for the pipe table row:
    # e.g. | image_tensor  | 1,224,224,3  | uFxp_8  | bitwidth 8, min 0.0000..., max 0.99609..., scale 0.00390625, offset 0.0 |
    row_match = re.search(r'\|\s*image_tensor\s*\|\s*([0-9,\s]+)\s*\|\s*([^\|]+?)\s*\|\s*([^\n\|]+)', text, flags=re.I)
    if row_match:
        dims_str = row_match.group(1).strip()
        type_field = row_match.group(2).strip()
        encoding_field = row_match.group(3).strip()
        try:
            dims = [int(x.strip()) for x in dims_str.split(',') if x.strip()]
            info['dims'] = dims
        except:
            pass
        info['type'] = type_field
        # find bitwidth
        bw_m = re.search(r'bitwidth\s*[:=]?\s*([0-9]+)', encoding_field, flags=re.I)
        if bw_m:
            info['bitwidth'] = int(bw_m.group(1))
        else:
            # sometimes type contains number e.g. uFxp_8
            t_m = re.search(r'[_\-](\d+)', type_field)
            if t_m:
                info['bitwidth'] = int(t_m.group(1))
        # scale / offset
        s_m = re.search(r'scale\s*[:=]?\s*([0-9eE\+\-\.]+)', encoding_field, flags=re.I)
        o_m = re.search(r'offset\s*[:=]?\s*([0-9eE\+\-\.]+)', encoding_field, flags=re.I)
        if s_m:
            info['scale'] = float(s_m.group(1))
        if o_m:
            info['offset'] = float(o_m.group(1))
        return info

    # Fallback 1: find the "Input Tensors" block then search for image_tensor lines
    block_match = re.search(r'Input Tensors(.*?)(?:Output Tensors|Total parameters|$)', text, flags=re.S|re.I)
    block = block_match.group(1) if block_match else text
    # find any line that mentions image_tensor
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        if 'image_tensor' in ln.lower():
            # take the current and next 2 lines and join them then try to parse dims/type/encoding
            window = " ".join(lines[i:i+3])
            # dims
            dims_m = re.search(r'(\d+(?:\s*,\s*\d+)+)', window)
            if dims_m:
                try:
                    info['dims'] = [int(x.strip()) for x in dims_m.group(1).split(',')]
                except:
                    pass
            # type
            t_m = re.search(r'(uFxp_\d+|iFxp_\d+|Float_\d+|UINT_\d+|INT_\d+|uFXP_\d+|f?Fxp_\d+)', window, flags=re.I)
            if t_m:
                info['type'] = t_m.group(1)
                tb_m = re.search(r'(\d+)', info['type'])
                if tb_m:
                    try:
                        info['bitwidth'] = int(tb_m.group(1))
                    except:
                        pass
            # scale/offset
            s_m = re.search(r'scale\s*[:=]?\s*([0-9eE\+\-\.]+)', window, flags=re.I)
            o_m = re.search(r'offset\s*[:=]?\s*([0-9eE\+\-\.]+)', window, flags=re.I)
            if s_m:
                info['scale'] = float(s_m.group(1))
            if o_m:
                info['offset'] = float(o_m.group(1))
            return info

    # Fallback 2: broad search for scale/offset anywhere and attempt to find dims nearby
    s_m = re.search(r'scale\s*[:=]?\s*([0-9eE\+\-\.]+)', text, flags=re.I)
    o_m = re.search(r'offset\s*[:=]?\s*([0-9eE\+\-\.]+)', text, flags=re.I)
    if s_m:
        info['scale'] = float(s_m.group(1))
    if o_m:
        info['offset'] = float(o_m.group(1))
    # dims: first occurrence of pattern like 1,224,224,3
    dims_m = re.search(r'(\b1\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\b)', text)
    if dims_m:
        try:
            info['dims'] = [int(x.strip()) for x in dims_m.group(1).split(',')]
        except:
            pass
    # type guess: search for 'Float_32' presence else look for 'uFxp' etc
    t_m = re.search(r'(Float_\d+|uFxp_\d+|iFxp_\d+|UINT_\d+|INT_\d+)', text, flags=re.I)
    if t_m:
        info['type'] = t_m.group(1)
        tb_m = re.search(r'(\d+)', info['type'])
        if tb_m:
            info['bitwidth'] = int(tb_m.group(1))
    return info

# ---------------------------
# Quantization function
# ---------------------------
def quantize(arr_0_1, scale, offset, bitwidth, signed=False):
    if scale is None:
        raise ValueError("scale is required for quantization but missing")
    q = np.round(arr_0_1 / float(scale) + float(offset)).astype(np.int64)
    if signed:
        lo = -(1 << (bitwidth - 1))
        hi = (1 << (bitwidth - 1)) - 1
    else:
        lo = 0
        hi = (1 << bitwidth) - 1
    q = np.clip(q, lo, hi)
    if bitwidth == 8:
        return q.astype(np.int8 if signed else np.uint8)
    elif bitwidth == 16:
        return q.astype(np.int16 if signed else np.uint16)
    else:
        return q.astype(np.int32)

# ---------------------------
# Float preprocessing
# ---------------------------
def preprocess_float(arr_0_1, mode='imagenet', mean=None, std=None):
    if mode == 'raw':
        return arr_0_1.astype(np.float32)
    if mode == 'imagenet':
        mean_arr = np.array([0.485,0.456,0.406], dtype=np.float32)
        std_arr  = np.array([0.229,0.224,0.225], dtype=np.float32)
    else:
        if mean is None or std is None:
            raise ValueError("custom requires mean/std")
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr  = np.array(std, dtype=np.float32)
    return ((arr_0_1 - mean_arr.reshape((1,1,3))) / std_arr.reshape((1,1,3))).astype(np.float32)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--dlcs', nargs='+', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--snpe_dlc_info', default='/opt/qairt/bin/x86_64-linux-clang/snpe-dlc-info')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--num_images', type=int, default=200)
    parser.add_argument('--float_mode', choices=['imagenet','raw','custom'], default='raw')
    parser.add_argument('--float_mean', nargs=3, type=float)
    parser.add_argument('--float_std', nargs=3, type=float)
    parser.add_argument('--device_prefix', default=None, help='If given, writes input_list_device.txt with device prefix')
    parser.add_argument('--layout_override', choices=['NHWC','NCHW','auto'], default='auto')
    parser.add_argument('--force_scale', type=float, default=None, help='If snpe-dlc-info lacks scale, force this')
    parser.add_argument('--force_offset', type=float, default=None, help='If snpe-dlc-info lacks offset, force this')
    parser.add_argument('--force_bitwidth', type=int, default=None)
    parser.add_argument('--resize_filter', choices=['bilinear','bicubic','nearest'], default='bilinear',
                        help='Interpolation filter used for resizing images (default: bilinear)')
    args = parser.parse_args()

    ds = Path(args.dataset_dir)
    if not ds.exists():
        print("ERROR: dataset_dir not found:", ds); sys.exit(1)
    img_files = sorted([p for p in ds.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    if len(img_files) == 0:
        print("ERROR: no images found in dataset_dir"); sys.exit(1)
    img_files = img_files[:args.num_images]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    snpe_bin = args.snpe_dlc_info

    # Pillow >=10 uses Image.Resampling.*; provide a backwards-compatible mapping
    try:
        RESAMPLING = Image.Resampling  # type: ignore[attr-defined]
        RES_BILINEAR = RESAMPLING.BILINEAR
        RES_BICUBIC = RESAMPLING.BICUBIC
        RES_NEAREST = RESAMPLING.NEAREST
    except AttributeError:
        RES_BILINEAR = Image.BILINEAR
        RES_BICUBIC = Image.BICUBIC
        RES_NEAREST = Image.NEAREST

    if args.resize_filter == 'bilinear':
        resize_interp = RES_BILINEAR
    elif args.resize_filter == 'bicubic':
        resize_interp = RES_BICUBIC
    else:
        resize_interp = RES_NEAREST

    for dlc in args.dlcs:
        dlc_path = Path(dlc)
        if not dlc_path.exists():
            # try relative to cwd
            dlc_path = (Path.cwd() / dlc).resolve()
            if not dlc_path.exists():
                print(f"ERROR: DLC not found: {dlc} (tried raw and cwd). Skipping."); continue
        print("\n=== DLC:", dlc_path.name)
        txt = run_snpe_dlc_info(snpe_bin, str(dlc_path))
        info = extract_input_info(txt)
        # allow force overrides
        if args.force_scale is not None:
            info['scale'] = args.force_scale
        if args.force_offset is not None:
            info['offset'] = args.force_offset
        if args.force_bitwidth is not None:
            info['bitwidth'] = args.force_bitwidth

        if not info['dims']:
            print("WARNING: could not parse dims; defaulting to [1,224,224,3]")
            info['dims'] = [1, args.size, args.size, 3]
        dims = info['dims']
        # layout detection
        if args.layout_override != 'auto':
            layout = args.layout_override
        else:
            if len(dims) >= 4 and dims[1] == 3:
                layout = 'NCHW'
            elif len(dims) >= 4 and dims[-1] == 3:
                layout = 'NHWC'
            else:
                layout = 'NHWC'

        print("Parsed -> dims:", info['dims'], "type:", info['type'], "bitwidth:", info['bitwidth'],
              "scale:", info['scale'], "offset:", info['offset'], "layout:", layout)

        out_dir = out_root / dlc_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        input_list = []
        for i, imgp in enumerate(img_files):
            im = Image.open(imgp).convert('RGB').resize((args.size, args.size), resize_interp)
            arr = np.array(im).astype(np.float32)  # H,W,C in 0..255
            arr_01 = (arr / 255.0).astype(np.float32)
            if info['type'] and ('Float' in info['type'] or 'Float_32' in info['type'] or info['type'].lower().startswith('float')):
                # float path
                if args.float_mode == 'custom':
                    proc = preprocess_float(arr_01, mode='custom', mean=args.float_mean, std=args.float_std)
                else:
                    proc = preprocess_float(arr_01, mode=args.float_mode)
                out_arr = proc
            else:
                # quantized path: need scale & offset & bitwidth
                if info['scale'] is None:
                    print("ERROR: DLC has no scale info and no --force_scale provided. Aborting for this DLC.")
                    sys.exit(1)
                bitw = info['bitwidth'] or (8 if '8' in (info.get('type') or '') else 16)
                # signedness: 'uFxp' -> unsigned; 'iFxp' or INT -> signed
                signed = False
                if info.get('type'):
                    if re.search(r'\b(iFxp|INT|INT_\d+|iFxp_\d+)\b', info['type'], flags=re.I):
                        signed = True
                    elif re.search(r'\buFxp\b|\bUINT\b', info['type'], flags=re.I):
                        signed = False
                    else:
                        # conservative default: unsigned for uFxp, signed for others containing 'int'
                        signed = bool(re.search(r'int', (info.get('type') or ''), flags=re.I))
                q = quantize(arr_01, info['scale'], info['offset'] if info['offset'] is not None else 0.0, bitw, signed=signed)
                out_arr = q
            # layout convert
            if layout == 'NCHW':
                if out_arr.ndim == 3:
                    out_arr = out_arr.transpose((2,0,1))
                # if float, shape becomes C,H,W; if quant, same transpose
            # write raw
            fname = f"{i:06d}_{imgp.stem}.raw"
            path = out_dir / fname
            out_arr.ravel().tofile(str(path))
            input_list.append(str(path.resolve()))
            written += 1

        # write host-side input_list.txt
        with open(out_dir / "input_list.txt", "w") as f:
            for p in input_list:
                f.write(p + "\n")
        # write device list if requested (device prefix used)
        if args.device_prefix:
            with open(out_dir / "input_list_device.txt", "w") as f:
                for p in input_list:
                    f.write(os.path.join(args.device_prefix, Path(p).name) + "\n")

        print(f"WROTE {written} inputs -> {out_dir}/")
    print("\nALL DONE.")

if __name__ == '__main__':
    main()
