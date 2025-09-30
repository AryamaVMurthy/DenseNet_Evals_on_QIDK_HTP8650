#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
import tempfile
import urllib.request
from pathlib import Path


def log(msg: str):
    print(f"[run_qidk_densenet] {msg}")


def check_tool(name: str):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required tool '{name}' not found in PATH")
    return path


def run(cmd, check=True, capture_output=False, text=True, env=None):
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text, env=env)


def _download_to_temp(url_or_path: str, suffix: str) -> Path:
    td = Path(tempfile.mkdtemp())
    dst = td / f"dataset{suffix}"
    # Support local file path
    if os.path.exists(url_or_path):
        shutil.copy2(url_or_path, dst)
    else:
        log(f"Downloading dataset: {url_or_path}")
        urllib.request.urlretrieve(url_or_path, dst)
    return dst


def download_imagenette(out_root: Path, url: str, subset: str, max_images: int) -> list[Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tgz_path = _download_to_temp(url, ".tgz")
        log(f"Extracting: {tgz_path}")
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(path=tgz_path.parent)
        # find subset directory (train/val)
        base = None
        for p in tgz_path.parent.iterdir():
            if p.is_dir() and p.name.startswith("imagenette2"):
                base = p
                break
        if base is None:
            raise RuntimeError("Could not locate extracted imagenette directory")
        subset_dir = base / subset
        if not subset_dir.exists():
            # some variants use 'val'/'train'; if missing, fallback to base
            subset_dir = base
        # Collect image paths
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = []
        for cls in sorted(subset_dir.iterdir()):
            if not cls.is_dir():
                continue
            for img in sorted(cls.rglob("*")):
                if img.suffix.lower() in exts:
                    imgs.append(img)
        if not imgs:
            raise RuntimeError(f"No images found under {subset_dir}")
        imgs = imgs[: max_images if max_images > 0 else len(imgs)]
        # Copy selected images into out_root for reproducibility
        selected = []
        for i, src in enumerate(imgs, 1):
            dst = out_root / f"img_{i:05d}{src.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            selected.append(dst)
        log(f"Prepared {len(selected)} images at {out_root}")
        return selected


def download_tiny_imagenet(out_root: Path, url: str | None, subset: str, max_images: int) -> list[Path]:
    """Download and prepare Tiny ImageNet-200.
    Default URL: http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """
    if not url:
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        zip_path = _download_to_temp(url, ".zip")
        log(f"Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=zip_path.parent)
        base = zip_path.parent / "tiny-imagenet-200"
        if not base.exists():
            # Try to find the folder if named differently
            cand = [p for p in zip_path.parent.iterdir() if p.is_dir() and p.name.startswith("tiny-imagenet")]
            if cand:
                base = cand[0]
        if not base.exists():
            raise RuntimeError("Could not locate extracted tiny-imagenet directory")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs: list[Path] = []
        subset = subset.lower()
        if subset.startswith("val"):
            val_dir = base / "val" / "images"
            if not val_dir.exists():
                raise RuntimeError(f"Expected {val_dir} not found")
            imgs = [p for p in sorted(val_dir.glob("*")) if p.suffix.lower() in exts]
        else:
            train_dir = base / "train"
            if not train_dir.exists():
                raise RuntimeError(f"Expected {train_dir} not found")
            for cls in sorted(train_dir.iterdir()):
                if not cls.is_dir():
                    continue
                img_dir = cls / "images"
                if img_dir.exists():
                    imgs.extend([p for p in sorted(img_dir.glob("*")) if p.suffix.lower() in exts])
        if not imgs:
            raise RuntimeError("No images found in tiny-imagenet")
        imgs = imgs[: max_images if max_images > 0 else len(imgs)]
        selected = []
        for i, src in enumerate(imgs, 1):
            dst = out_root / f"img_{i:05d}{src.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            selected.append(dst)
        log(f"Prepared {len(selected)} images at {out_root}")
        return selected


def collect_images_from_dir(images_dir: Path, max_images: int) -> list[Path]:
    if not images_dir.exists():
        raise RuntimeError(f"Images directory not found: {images_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in exts]
    if not imgs:
        raise RuntimeError(f"No images found under {images_dir}")
    if max_images > 0:
        imgs = imgs[:max_images]
    log(f"Using {len(imgs)} existing image(s) from {images_dir}")
    return imgs


def ensure_pillow():
    try:
        import PIL  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        log("Pillow not found. Installing...")
        run([sys.executable, "-m", "pip", "install", "--quiet", "pillow"], check=True)


def preprocess_to_raw(images: list[Path], out_root: Path, tensor_name: str, height: int, width: int) -> tuple[Path, list[Path]]:
    ensure_pillow()
    from PIL import Image
    raws_dir = out_root / "raw"
    raws_dir.mkdir(parents=True, exist_ok=True)
    input_list = out_root / "input_list.txt"
    raw_paths = []
    with input_list.open("w") as f:
        for i, img_path in enumerate(images, 1):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize((width, height), Image.BILINEAR)
                # Convert to uint8 NHWC, add batch dim=1
                arr = bytearray(im.tobytes())
                # Save as .raw (1xHxWx3 contiguous)
                raw_path = raws_dir / f"sample_{i:05d}.raw"
                with raw_path.open("wb") as rf:
                    rf.write(arr)
            f.write(f"{tensor_name}:={raw_path.name}\n")
            raw_paths.append(raw_path)
    # write manifest mapping raw -> source image
    manifest = out_root / "input_manifest.csv"
    with manifest.open("w") as mf:
        mf.write("index,raw,src\n")
        for i, (rp, src) in enumerate(zip(raw_paths, images)):
            mf.write(f"{i},{rp.name},{src}\n")
    log(f"Wrote {len(raw_paths)} raw files and input list at {out_root}")
    return input_list, raw_paths


def preprocess_to_raw_float(images: list[Path], out_root: Path, tensor_name: str, height: int, width: int) -> tuple[Path, list[Path]]:
    """Create float32 NHWC inputs and a corresponding input_list for tools that expect FLOAT input files.
    Values are scaled to [0,1] by dividing by 255.0.
    """
    ensure_pillow()
    from PIL import Image
    raws_dir = out_root / "raw_f32"
    raws_dir.mkdir(parents=True, exist_ok=True)
    input_list = out_root / "input_list_float.txt"
    raw_paths = []
    import numpy as np  # local import to avoid global dependency if unused
    with input_list.open("w") as f:
        for i, img_path in enumerate(images, 1):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize((width, height), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC in [0,1]
                arr = arr.reshape((1, height, width, 3))  # NHWC with batch=1
                raw_path = raws_dir / f"sample_{i:05d}_f32.raw"
                arr.tofile(raw_path)
            f.write(f"{tensor_name}:={raw_path.name}\n")
            raw_paths.append(raw_path)
    # write manifest for float inputs
    manifest = out_root / "input_manifest_float.csv"
    with manifest.open("w") as mf:
        mf.write("index,raw,src\n")
        for i, (rp, src) in enumerate(zip(raw_paths, images)):
            mf.write(f"{i},{rp.name},{src}\n")
    log(f"Wrote {len(raw_paths)} float32 raw files and input list at {out_root}")
    return input_list, raw_paths


def preprocess_to_raw_u16(images: list[Path], out_root: Path, tensor_name: str, height: int, width: int,
                          scale: float, offset: float, signed: bool = False) -> tuple[Path, list[Path]]:
    """Create 16-bit fixed-point NHWC inputs from images using scale/offset.
    If signed=True, uses int16; else uint16.
    Assumes images are standard 0..255 RGB; converts to [0,1] then quantizes.
    """
    ensure_pillow()
    from PIL import Image
    raws_dir = out_root / "raw_u16"
    raws_dir.mkdir(parents=True, exist_ok=True)
    input_list = out_root / "input_list_u16.txt"
    raw_paths = []
    import numpy as np
    dt = np.int16 if signed else np.uint16
    with input_list.open("w") as f:
        for i, img_path in enumerate(images, 1):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize((width, height), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0  # [0,1]
                q = np.round(arr / float(scale) + float(offset))
                if signed:
                    q = np.clip(q, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
                else:
                    q = np.clip(q, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                q = q.reshape((1, height, width, 3))
                raw_path = raws_dir / f"sample_{i:05d}_u16.raw"
                q.tofile(raw_path)
            f.write(f"{tensor_name}:={raw_path.name}\n")
            raw_paths.append(raw_path)
    manifest = out_root / "input_manifest_u16.csv"
    with manifest.open("w") as mf:
        mf.write("index,raw,src\n")
        for i, (rp, src) in enumerate(zip(raw_paths, images)):
            mf.write(f"{i},{rp.name},{src}\n")
    log(f"Wrote {len(raw_paths)} uint16 raw files and input list at {out_root}")
    return input_list, raw_paths


def read_context_info(context_json: Path):
    with context_json.open() as f:
        j = json.load(f)
    graphs = j["info"]["graphs"]
    if not graphs:
        raise RuntimeError("No graphs in context json")
    g0 = graphs[0]["info"]
    inp = g0["graphInputs"][0]["info"]
    tensor_name = inp["name"]
    dims = inp["dimensions"]
    dtype = inp["dataType"]
    quant = inp.get("quantizeParams", {})
    ctx_arch = j["info"].get("contextMetadata", {}).get("info", {}).get("dspArch", None)
    return {
        "tensor_name": tensor_name,
        "dims": dims,
        "dtype": dtype,
        "quant": quant,
        "dspArch": ctx_arch,
    }


def verify_qnn_sdk(qnn_sdk_root: Path, require_throughput: bool = True):
    req = {
        "net_run": qnn_sdk_root / "bin/aarch64-android/qnn-net-run",
        "throughput": qnn_sdk_root / "bin/aarch64-android/qnn-throughput-net-run",
        "lib_htp": qnn_sdk_root / "lib/aarch64-android/libQnnHtp.so",
        "lib_prepare": qnn_sdk_root / "lib/aarch64-android/libQnnHtpPrepare.so",
        "lib_stub_v75": qnn_sdk_root / "lib/aarch64-android/libQnnHtpV75Stub.so",
        "skel_v75": qnn_sdk_root / "lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so",
        "prof_viewer": qnn_sdk_root / "bin/x86_64-linux-clang/qnn-profile-viewer",
        "prof_reader": qnn_sdk_root / "lib/x86_64-linux-clang/libQnnHtpProfilingReader.so",
        "thr_ext": qnn_sdk_root / "lib/aarch64-android/libQnnHtpNetRunExtensions.so",
    }
    missing = []
    for k, p in req.items():
        if (k == "throughput" or k == "thr_ext") and not require_throughput:
            continue
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise RuntimeError("Missing required QNN SDK files:\n" + "\n".join(missing))
    return req


def adb_push(local: Path, remote: str):
    run(["adb", "push", str(local), remote])


def adb_shell(cmd: str):
    return run(["adb", "shell", cmd])


def main():
    ap = argparse.ArgumentParser(description="End-to-end runner for densenet-w8a8 on QIDK SM8650 (Android)")
    ap.add_argument("--qnn_sdk_root", type=Path, default=os.environ.get("QNN_SDK_ROOT", ""), help="Path to QNN SDK root")
    ap.add_argument("--context", type=Path, default=Path("densenet-w8a8.context.json"), help="Path to context json")
    ap.add_argument("--binary", type=Path, default=Path("densenet-w8a8.bin"), help="Path to offline context binary")
    ap.add_argument("--dataset", default="imagenette", choices=["imagenette", "tiny-imagenet"], help="Dataset to download if --images_dir not provided")
    ap.add_argument("--dataset_url", default=None, help="Override dataset URL (optional)")
    ap.add_argument("--images_dir", type=Path, default=None, help="Use images from this directory instead of downloading")
    ap.add_argument("--subset", default="val", help="Subset under dataset (val/train)")
    ap.add_argument("--out", type=Path, default=Path("inputs_imagenette"), help="Local output directory for preprocessed inputs")
    ap.add_argument("--max_images", type=int, default=50, help="Number of images to process (0 = all)")
    ap.add_argument("--device_dir", default="/data/local/tmp/qnn/densenet", help="Directory on device for files")
    ap.add_argument("--profiling_level", default="backend", choices=["basic", "detailed", "client", "backend"], help="qnn-net-run profiling level")
    ap.add_argument("--htp_prof_level", default="detailed", choices=["detailed", "linting"], help="HTP backend profiling level (used when --profiling_level=backend)")
    ap.add_argument("--chrometrace", action="store_true", help="Also generate Chrometrace timeline (disabled by default)")
    ap.add_argument("--perf_profile", default="high_performance", help="HTP perf profile")
    ap.add_argument("--backend", default="auto", choices=["auto","htp","cpu"], help="Execution backend: auto picks CPU for FLOAT I/O, HTP otherwise")
    ap.add_argument("--run_throughput", action="store_true", help="Also run qnn-throughput-net-run benchmark")
    ap.add_argument("--thr_loops", type=int, default=50, help="Throughput loop count")
    args = ap.parse_args()

    if not args.qnn_sdk_root:
        raise RuntimeError("--qnn_sdk_root not provided and QNN_SDK_ROOT env not set")
    qnn_sdk_root = args.qnn_sdk_root.resolve()

    check_tool("adb")

    # 1) Read context for input spec
    info = read_context_info(args.context)
    tensor_name = info["tensor_name"]
    dims = info["dims"]
    dtype = info["dtype"]
    arch = info.get("dspArch")
    if dtype != "QNN_DATATYPE_UFIXED_POINT_8":
        log(f"WARNING: Expected uint8 input, but context dtype is {dtype}")
    if len(dims) != 4:
        raise RuntimeError(f"Unexpected input rank: {dims}")
    b, h, w, c = dims
    if not (b == 1 and c == 3):
        log(f"WARNING: Unusual dims {dims}; proceeding anyway")
    log(f"Context input: name={tensor_name} dims={dims} dtype={dtype} arch={arch}")

    # 2) Download and preprocess dataset
    # Prefer user-provided images_dir if given
    if args.images_dir:
        imgs = collect_images_from_dir(args.images_dir, args.max_images)
    else:
        # If out/images already has images, reuse them (skip download)
        cached_dir = args.out / "images"
        existing = None
        try:
            existing = collect_images_from_dir(cached_dir, args.max_images)
        except Exception:
            existing = None
        if existing:
            imgs = existing
            log("Found existing images in output folder; skipping download.")
        else:
            if args.dataset == "imagenette":
                ds_url = args.dataset_url or "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
                imgs = download_imagenette(cached_dir, ds_url, args.subset, args.max_images)
            else:
                imgs = download_tiny_imagenet(cached_dir, args.dataset_url, args.subset, args.max_images)
    # Always prepare multiple input types; select based on context dtype
    input_list_u8, raw_paths_u8 = preprocess_to_raw(imgs, args.out, tensor_name, h, w)
    input_list_f32, raw_paths_f32 = preprocess_to_raw_float(imgs, args.out, tensor_name, h, w)
    # If input is 16-bit fixed point, quantize accordingly
    input_list_u16 = raw_paths_u16 = None
    qparams = info.get("quant", {}).get("scaleOffset", {})
    in_scale = qparams.get("scale")
    in_offset = qparams.get("offset", 0)
    dtype_str = str(dtype)
    if "UFIXED_POINT_16" in dtype_str or "SFIXED_POINT_16" in dtype_str:
        input_list_u16, raw_paths_u16 = preprocess_to_raw_u16(
            imgs, args.out, tensor_name, h, w, in_scale or 1.0, in_offset or 0.0, signed=("S" in dtype_str)
        )
        selected_input_list = input_list_u16
        selected_raw_paths = raw_paths_u16
        log("Using uint16 inputs based on context dtype")
    elif dtype_str.startswith("QNN_DATATYPE_FLOAT"):
        selected_input_list = input_list_f32
        selected_raw_paths = raw_paths_f32
        log("Using float32 inputs based on context dtype")
    else:
        selected_input_list = input_list_u8
        selected_raw_paths = raw_paths_u8
        log("Using uint8 inputs based on context dtype")

    # 3) Verify QNN SDK files
    req = verify_qnn_sdk(qnn_sdk_root, require_throughput=True)
    log("QNN SDK files verified.")

    # 4) Push to device
    adb_shell(f"mkdir -p {args.device_dir}")
    adb_push(args.binary, args.device_dir + "/")
    adb_push(selected_input_list, args.device_dir + "/")
    for rp in selected_raw_paths:
        adb_push(rp, args.device_dir + "/")
    # If throughput is requested, also push the alternate dtype inputs (float32)
    if args.run_throughput:
        adb_push(input_list_f32, args.device_dir + "/")
        for rp in raw_paths_f32:
            adb_push(rp, args.device_dir + "/")
    # Tools and libs
    adb_push(req["net_run"], args.device_dir + "/")
    # Select backend
    dtype_str = str(dtype)
    if args.backend == "auto":
        selected_backend = "cpu" if dtype_str.startswith("QNN_DATATYPE_FLOAT") else "htp"
    else:
        selected_backend = args.backend
    log(f"Selected backend: {selected_backend}")

    backend_lib = None
    if selected_backend == "htp":
        # HTP libs
        adb_push(req["lib_htp"], args.device_dir + "/")
        adb_push(req["lib_prepare"], args.device_dir + "/")
        # Choose correct HTP stub/skel based on context dspArch
        arch_map_stub = {68: "libQnnHtpV68Stub.so", 69: "libQnnHtpV69Stub.so", 73: "libQnnHtpV73Stub.so", 75: "libQnnHtpV75Stub.so", 79: "libQnnHtpV79Stub.so"}
        try:
            arch_int = int(arch) if arch is not None else 75
        except Exception:
            arch_int = 75
        stub_name = arch_map_stub.get(arch_int, "libQnnHtpV75Stub.so")
        stub_path = qnn_sdk_root / "lib/aarch64-android" / stub_name
        skel_path = qnn_sdk_root / f"lib/hexagon-v{arch_int}" / "unsigned" / f"libQnnHtpV{arch_int}Skel.so"
        if not stub_path.exists():
            log(f"WARNING: Stub {stub_name} not found; falling back to V75")
            stub_path = qnn_sdk_root / "lib/aarch64-android/libQnnHtpV75Stub.so"
        if not skel_path.exists():
            log(f"WARNING: Skel for v{arch_int} not found; falling back to v75")
            skel_path = qnn_sdk_root / "lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"
        adb_push(stub_path, args.device_dir + "/")
        adb_push(skel_path, args.device_dir + "/")
        # Backend extension for profiling
        adb_push(req["thr_ext"], args.device_dir + "/")
        backend_lib = "libQnnHtp.so"
    else:
        # CPU backend
        cpu_lib = qnn_sdk_root / "lib/aarch64-android/libQnnCpu.so"
        adb_push(cpu_lib, args.device_dir + "/")
        backend_lib = "libQnnCpu.so"
    # HTP backend extension (needed for backend-level profiling logs)
    adb_push(req["thr_ext"], args.device_dir + "/")
    if args.run_throughput:
        adb_push(req["throughput"], args.device_dir + "/")

    # 5) Run qnn-net-run on device
    # Prepare backend extension config for HTP profiling logs
    backend_ext_cfg = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": "htp_profile_config.json",
        }
    }
    local_backend_ext = Path("backend_extension_config.json")
    with local_backend_ext.open("w") as f:
        json.dump(backend_ext_cfg, f, indent=2)
    # HTP profiling config (detailed or linting)
    htp_profile_cfg = {"devices": [{"profiling_level": args.htp_prof_level}]}
    local_htp_profile = Path("htp_profile_config.json")
    with local_htp_profile.open("w") as f:
        json.dump(htp_profile_cfg, f, indent=2)
    if selected_backend == "htp":
        adb_push(local_backend_ext, args.device_dir + "/")
        adb_push(local_htp_profile, args.device_dir + "/")

    # Clean remote output directory to avoid stale results
    try:
        adb_shell(f"rm -rf {args.device_dir}/output")
    except Exception:
        pass

    run_cmd = (
        f"cd {args.device_dir} && "
        f"export LD_LIBRARY_PATH={args.device_dir}:/system/lib64:/vendor/lib64:$LD_LIBRARY_PATH && "
        + (f"export ADSP_LIBRARY_PATH=\\\"{args.device_dir}:/dsp/cdsp:/system/lib/rfsa/adsp:/vendor/lib/rfsa/adsp:/dsp\\\" && " if selected_backend=="htp" else "") +
        f"./qnn-net-run --retrieve_context {args.binary.name} --backend {backend_lib} "
        f"--input_list {selected_input_list.name} --use_native_input_files "
        f"--perf_profile {args.perf_profile} "
        f"--profiling_level {('detailed' if (selected_backend=='cpu' and args.profiling_level=='backend') else args.profiling_level)} "
        + ((f"--config_file {local_backend_ext.name} ") if (selected_backend=="htp" and args.profiling_level=="backend") else "") +
        f"--validate_binary --log_level error --output_dir output"
    )
    adb_shell(run_cmd)

    # 6) Pull outputs back
    host_out = Path("output_android")
    if host_out.exists():
        shutil.rmtree(host_out)
    host_out.mkdir(parents=True, exist_ok=True)
    run(["adb", "pull", f"{args.device_dir}/output", str(host_out)])
    # Look for HTP profiling log produced by backend extension
    output_dir = host_out / "output"
    prof_json = output_dir / "profiling.json"
    htp_logs = sorted(output_dir.glob("qnn-profiling-data_*.log"))
    if prof_json.exists():
        # Non-HTP reader path (keep for completeness)
        csv_out = host_out / "profile.csv"
        # Suppress profile-viewer stdout; we will print a concise summary later
        subprocess.run([
            str(req["prof_viewer"]),
            "--input_log",
            str(prof_json),
            "--output",
            str(csv_out),
        ], check=True, capture_output=True, text=True)
        log(f"Profile CSV (generic): {csv_out}")
    elif htp_logs:
        csv_out = host_out / "profile_htp.csv"
        subprocess.run([
            str(req["prof_viewer"]),
            "--input_log",
            str(htp_logs[0]),
            "--reader",
            str(req["prof_reader"]),
            "--output",
            str(csv_out),
        ], check=True, capture_output=True, text=True)
        log(f"Profile CSV (HTP): {csv_out}")
        # Also create Chrometrace JSON timeline if requested
        if args.chrometrace:
            qnn_root_env = os.environ.get("QNN_SDK_ROOT")
            if qnn_root_env:
                qnn_root = Path(qnn_root_env)
            else:
                viewer_path = Path(str(req["prof_viewer"]))
                # .../bin/x86_64-linux-clang/qnn-profile-viewer -> root is three parents up
                qnn_root = viewer_path.parent.parent.parent
            chrometrace_reader = qnn_root / "lib/x86_64-linux-clang/libQnnChrometraceProfilingReader.so"
            if chrometrace_reader.is_file():
                trace_out = host_out / "chrometrace.json"
                subprocess.run([
                    str(req["prof_viewer"]),
                    "--input_log",
                    str(htp_logs[0]),
                    "--reader",
                    str(chrometrace_reader),
                    "--output",
                    str(trace_out),
                ], check=True, capture_output=True, text=True)
                log(f"Chrometrace: {trace_out}")
            else:
                log(f"Chrometrace reader not found at {chrometrace_reader}; skipping timeline generation.")
    else:
        log("No profiling logs found in output; ensure backend profiling is enabled.")

    # Generate a concise markdown report
    def _parse_profile_report(txt_path: Path):
        try:
            txt = txt_path.read_text(errors="ignore")
        except Exception:
            return {}
        import re, statistics
        qtimes = [int(x) for x in re.findall(r"QNN \(execute\) time\s*:\s*(\d+)\s*us", txt)]
        atimes = [int(x) for x in re.findall(r"Accelerator \(execute\) time\s*:\s*(\d+)\s*us", txt)]
        summary = {}
        if qtimes:
            summary.update({
                "qnn_avg_us": statistics.mean(qtimes),
                "qnn_p50_us": statistics.median(qtimes),
                "qnn_min_us": min(qtimes),
                "qnn_max_us": max(qtimes),
                "ips": 1e6/ statistics.mean(qtimes)
            })
        if atimes:
            summary.update({
                "acc_avg_us": statistics.mean(atimes),
                "acc_p50_us": statistics.median(atimes),
                "acc_min_us": min(atimes),
                "acc_max_us": max(atimes),
            })
        return summary

    # Read execution metadata for count
    exec_meta = output_dir / "execution_metadata.yaml"
    inferences = None
    if exec_meta.exists():
        for line in exec_meta.read_text().splitlines():
            if line.strip().startswith("inferences_completed:"):
                try:
                    inferences = int(line.split(":")[-1])
                except Exception:
                    pass
                break

    report_lines = []
    report_lines.append(f"# QNN Run Report\n")
    report_lines.append(f"- Device dir: `{args.device_dir}`\n")
    report_lines.append(f"- Perf profile: `{args.perf_profile}` | HTP profiling: `{args.htp_prof_level}`\n")
    if inferences is not None:
        report_lines.append(f"- Inferences completed: {inferences}\n")
    # Profiling summary (print to terminal and include in report)
    if htp_logs:
        # Prefer parsing raw log; if not available, parse CSV header
        raw_log = output_dir / "qnn-profiling-data_0.log"
        prof_summary = _parse_profile_report(raw_log if raw_log.exists() else (host_out / "profile_htp.csv"))
        if prof_summary:
            log("Final profiling summary (per-inference):")
            for k in ["qnn_avg_us","qnn_p50_us","qnn_min_us","qnn_max_us","ips","acc_avg_us","acc_p50_us","acc_min_us","acc_max_us"]:
                if k in prof_summary:
                    val = prof_summary[k]
                    unit = "img/s" if k == "ips" else "us"
                    print(f"  {k}: {val:.2f} {unit}")
        if prof_summary:
            report_lines.append("## Latency Summary (us)\n")
            for k, v in prof_summary.items():
                if k == "ips":
                    report_lines.append(f"- {k}: {v:.2f}\n")
                else:
                    report_lines.append(f"- {k}: {v:.2f}\n")
    # Link artifacts
    report_lines.append("## Artifacts\n")
    if (host_out/"profile_htp.csv").exists():
        report_lines.append(f"- HTP CSV: `output_android/profile_htp.csv`\n")
    if (host_out/"chrometrace.json").exists():
        report_lines.append(f"- Chrometrace: `output_android/chrometrace.json`\n")
    if (host_out/"output").exists():
        report_lines.append(f"- Raw outputs: `output_android/output/Result_*/class_logits.raw`\n")
    # If external accuracy report exists, reference it
    if (host_out.parent/"accuracy_report.csv").exists():
        report_lines.append(f"- Accuracy report: `output_android/accuracy_report.csv`\n")
    # If summary.json exists, include accuracy and latency
    summary_path = host_out.parent/"summary.json"
    if summary_path.exists():
        report_lines.append(f"- Summary JSON: `output_android/summary.json`\n")
        try:
            s = json.loads(summary_path.read_text())
            acc = s.get("accuracy", {})
            prof = s.get("profiling", {})
            report_lines.append("\n## Accuracy\n")
            if acc:
                if "count" in acc:
                    report_lines.append(f"- Samples: {acc['count']}\n")
                if "top1" in acc:
                    report_lines.append(f"- Top-1: {acc['top1']*100:.2f}%\n")
                if "top5" in acc:
                    report_lines.append(f"- Top-5: {acc['top5']*100:.2f}%\n")
                if "top10" in acc:
                    report_lines.append(f"- Top-10: {acc['top10']*100:.2f}%\n")
            report_lines.append("\n## Latency (from summary)\n")
            for k in ["qnn_execute_us_avg","qnn_execute_us_p50","qnn_execute_us_p90","qnn_execute_us_p95","qnn_execute_us_p99","qnn_execute_us_min","qnn_execute_us_max","ips"]:
                if k in prof:
                    v = prof[k]
                    unit = "img/s" if k == "ips" else "us"
                    report_lines.append(f"- {k}: {v:.2f} {unit}\n")
        except Exception:
            pass

    (host_out.parent/"report.md").write_text("".join(report_lines))
    log(f"Report: {host_out.parent/'report.md'}")

    # 7) Optional throughput benchmark
    if args.run_throughput:
        # Create a local config and push
        cfg = {
            "backends": [
                {
                    "backendName": "htp",
                    "backendPath": "libQnnHtp.so",
                    "profilingLevel": "OFF",
                    "backendExtensions": "libQnnHtpNetRunExtensions.so",
                    "perfProfile": args.perf_profile,
                }
            ],
            "models": [
                {
                    "modelName": "densenet",
                    "modelPath": args.binary.name,
                    "loadFromCachedBinary": True,
                    "inputPath": (float_input_list.name if float_input_list else input_list.name),
                    "inputDataType": "FLOAT",
                    "outputPath": "thr_out",
                    "outputDataType": "FLOAT_ONLY",
                    "saveOutput": "NONE"
                }
            ],
            "contexts": [{"contextName": "htp_ctx"}],
            "testCase": {
                "iteration": args.thr_loops,
                "threads": [
                    {
                        "threadName": "t0",
                        "backend": "htp",
                        "context": "htp_ctx",
                        "model": "densenet",
                        "interval": 0,
                        "loopUnit": "count",
                        "loop": args.thr_loops,
                    }
                ],
            },
        }
        local_cfg = Path("throughput_config.json")
        with local_cfg.open("w") as f:
            json.dump(cfg, f, indent=2)
        adb_push(local_cfg, args.device_dir + "/")
        thr_cmd = (
            f"cd {args.device_dir} && "
            f"export LD_LIBRARY_PATH={args.device_dir}:/system/lib64:/vendor/lib64:$LD_LIBRARY_PATH && "
            f"export ADSP_LIBRARY_PATH=\"{args.device_dir}:/dsp/cdsp:/system/lib/rfsa/adsp:/vendor/lib/rfsa/adsp:/dsp\" && "
            f"./qnn-throughput-net-run --config {local_cfg.name} --output thr_results.json"
        )
        try:
            adb_shell(thr_cmd)
            run(["adb", "pull", f"{args.device_dir}/thr_results.json", str(host_out)])
            log(f"Throughput results: {host_out / 'thr_results.json'}")
        except subprocess.CalledProcessError as e:
            log("qnn-throughput-net-run failed; falling back to qnn-net-run --num_inferences")
            # Fallback: use qnn-net-run client profiling for N inferences
            fallback_cmd = (
                f"cd {args.device_dir} && "
                f"export LD_LIBRARY_PATH={args.device_dir}:/system/lib64:/vendor/lib64:$LD_LIBRARY_PATH && "
                f"export ADSP_LIBRARY_PATH=\"{args.device_dir}:/dsp/cdsp:/system/lib/rfsa/adsp:/vendor/lib/rfsa/adsp:/dsp\" && "
                f"./qnn-net-run --retrieve_context {args.binary.name} --backend libQnnHtp.so "
                f"--input_list {input_list.name} --use_native_input_files --perf_profile {args.perf_profile} "
                f"--profiling_level client --num_inferences {args.thr_loops} --output_dir output_tnr"
            )
            # Capture output for reporting
            res = subprocess.run(["adb", "shell", fallback_cmd], capture_output=True, text=True)
            (host_out / "tnr_fallback.log").write_text(res.stdout + "\n" + res.stderr)
            log(f"Fallback throughput log: {host_out / 'tnr_fallback.log'}")

    log("Done.")


if __name__ == "__main__":
    main()
