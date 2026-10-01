#!/usr/bin/env python3
"""Benchmark SLIC, SICLE and SICLE irregular without retaining intermediates."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "assets" / "INPUT"
SVG_ROOT = ROOT / "assets" / "SVG"
RESULTS_DIR = ROOT / "metrics" / "results"
TEMP_ROOT = ROOT / "assets" / ".benchmark_tmp"

METHODS = ("slic", "sicle_comp", "sicle_irreg")
SUPERPIXELS = (500, 2000, 4000)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDS = (
    "image", "method", "superpixels", "status", "saved_svg",
    "processing_time_s", "mse", "psnr", "ssim", "lpips", "error",
)


def dependencies():
    try:
        import cairosvg
        import lpips
        import numpy as np
        import torch
        from PIL import Image
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as exc:
        raise SystemExit(
            "Dependência ausente: %s\nInstale com:\n"
            "python3 -m pip install -r metrics/requirements.txt" % exc.name
        ) from exc
    return cairosvg, lpips, np, torch, Image, peak_signal_noise_ratio, structural_similarity


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("comando falhou:\n" + completed.stdout[-2000:])


def segment(method: str, image: Path, label: Path, superpixels: int) -> None:
    if method == "slic":
        command = [str(ROOT / "scripts" / "RunSlic"), "--img", str(image),
                   "--out", str(label), "--n", str(superpixels), "--compactness", "10"]
    else:
        executable = "RunSICLECOMP" if method == "sicle_comp" else "RunSICLEIRREG"
        command = [str(ROOT / "bin" / executable), "--img", str(image),
                   "--out", str(label), "--nf", str(superpixels), "--n0", str(superpixels * 2)]
    run(command)


def vectorize(method: str, image: Path, label: Path, output_base: Path) -> Path:
    run([str(ROOT / "bin" / "RunVILLI"), "--img", str(image),
         "--labels", str(label), "--out", str(output_base),
         "--method", "slic" if method == "slic" else "sicle"])
    svg = output_base.with_suffix(".svg")
    if not svg.is_file():
        raise RuntimeError(f"SVG não gerado: {svg}")
    return svg


def metric_values(original: Path, svg: Path, modules, lpips_model):
    cairosvg, _, np, torch, Image, psnr_fn, ssim_fn = modules
    with Image.open(original) as source:
        source = source.convert("RGB")
        width, height = source.size
        reference = np.array(source, dtype=np.uint8, copy=True)

    rendered_bytes = cairosvg.svg2png(url=str(svg), output_width=width, output_height=height)
    from io import BytesIO
    with Image.open(BytesIO(rendered_bytes)) as image:
        rendered = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)

    reference_float = reference.astype(np.float32) / 255.0
    rendered_float = rendered.astype(np.float32) / 255.0
    mse = float(np.mean((reference_float - rendered_float) ** 2))
    psnr = float(psnr_fn(reference_float, rendered_float, data_range=1.0))
    ssim = float(ssim_fn(reference_float, rendered_float, channel_axis=2, data_range=1.0))

    tensors = [
        torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1.0)
        for array in (reference, rendered)
    ]
    with torch.no_grad():
        lpips_score = float(lpips_model(*tensors).item())
    return mse, psnr, ssim, lpips_score


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["image"] != "__mean__"]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    successful = [row for row in rows if row["status"] == "ok"]
    mean = {"image": "__mean__", "method": rows[0]["method"], "superpixels": rows[0]["superpixels"],
            "status": "summary", "saved_svg": "", "error": ""}
    for field in ("processing_time_s", "mse", "psnr", "ssim", "lpips"):
        mean[field] = sum(float(row[field]) for row in successful) / len(successful) if successful else ""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean)


def one_configuration(method: str, superpixels: int, images: list[Path],
                      save_limit: int, modules, lpips_model) -> None:
    csv_path = RESULTS_DIR / f"{method}_{superpixels}.csv"
    rows = read_rows(csv_path)
    completed = {row["image"] for row in rows if row["status"] == "ok"}
    saved = sum(row["saved_svg"] == "true" for row in rows if row["status"] == "ok")
    output_dir = SVG_ROOT / method / str(superpixels)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = TEMP_ROOT / method / str(superpixels)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(images, 1):
        if image.name in completed:
            continue
        print(f"[{method}/{superpixels}] {index}/{len(images)} {image.name}", flush=True)
        label = temp_dir / f"{image.stem}.png"
        output_base = temp_dir / image.stem
        temporary_svg = output_base.with_suffix(".svg")
        start = time.perf_counter()
        row = {"image": image.name, "method": method, "superpixels": superpixels,
               "status": "error", "saved_svg": "false", "processing_time_s": "",
               "mse": "", "psnr": "", "ssim": "", "lpips": "", "error": ""}
        try:
            segment(method, image, label, superpixels)
            svg = vectorize(method, image, label, output_base)
            elapsed = time.perf_counter() - start
            mse, psnr, ssim, lpips_score = metric_values(image, svg, modules, lpips_model)
            row.update(status="ok", processing_time_s=f"{elapsed:.6f}", mse=f"{mse:.8f}",
                       psnr=f"{psnr:.8f}", ssim=f"{ssim:.8f}", lpips=f"{lpips_score:.8f}")
            if saved < save_limit:
                destination = output_dir / f"{image.stem}.svg"
                shutil.move(str(svg), destination)
                row["saved_svg"] = "true"
                saved += 1
        except Exception as exc:
            row["error"] = str(exc).replace("\n", " ")[:2000]
        finally:
            label.unlink(missing_ok=True)
            temporary_svg.unlink(missing_ok=True)
            output_base.with_suffix(".gri").unlink(missing_ok=True)

        rows = [previous for previous in rows if previous["image"] != image.name]
        rows.append(row)
        write_rows(csv_path, rows)  # checkpoint: safe to resume after interruption

    shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--superpixels", nargs="+", type=int, choices=SUPERPIXELS,
                        default=SUPERPIXELS)
    parser.add_argument("--limit", type=int, default=None,
                        help="processa somente as primeiras N imagens (teste)")
    parser.add_argument("--save-limit", type=int, default=30)
    args = parser.parse_args()

    modules = dependencies()
    _, lpips, _, torch, _, _, _ = modules
    torch.set_grad_enabled(False)
    lpips_model = lpips.LPIPS(net="alex").eval()

    images = sorted(path for path in INPUT_DIR.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"Nenhuma imagem em {INPUT_DIR}")
    if args.limit is not None:
        images = images[:args.limit]

    for method in args.methods:
        for superpixels in args.superpixels:
            one_configuration(method, superpixels, images, args.save_limit,
                              modules, lpips_model)


if __name__ == "__main__":
    main()
