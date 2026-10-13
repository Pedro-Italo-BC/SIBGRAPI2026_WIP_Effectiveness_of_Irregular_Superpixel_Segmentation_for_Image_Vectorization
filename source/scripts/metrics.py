#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from importlib import import_module
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "assets" / "INPUT"
TEMP_ROOT = ROOT / "assets" / ".benchmark_tmp"
LABEL_ROOT = ROOT / "assets" / "labels"
SICLE_BIN = ROOT / "segmentation" / "SICLE" / "bin"
LSC_BIN = ROOT / "segmentation" / "LSC" / "bin" / "RunLSC"
SH_PYBUILD = ROOT / "segmentation" / "SH" / "pybuild"

SVG_ROOT = ROOT / "assets" / "SVG"
RESULTS_DIR_SVG = ROOT / "metrics" / "results_svg"

MEAN_ROOT = ROOT / "assets" / "MEAN" 
RESULTS_DIR_MEAN = ROOT / "metrics" / "results_mean"

METHODS = ("slic", "sicle_comp", "sicle_irreg", "LSC", "SH", "SEEDS")
SUPERPIXELS = (500, 2000, 4000)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRIC_MODES = ("svg", "mean", "both")
FIELDS_SVG: list[str] = [
    "image", "method", "superpixels", "status", "saved_svg",
    "processing_time_s", "mse", "psnr", "ssim", "lpips", "br", "ue", "error",
]
FIELDS_MEAN: list[str] = [
    "image", "method", "superpixels", "status", "saved_mean_img",
    "processing_time_s", "mse", "psnr", "ssim", "lpips", "br", "ue", "error",
]

def parse_float(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError(f"Esperava texto nnumérico: {type(value).__name__}")
    return float(value)


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
    label.parent.mkdir(parents=True, exist_ok=True)

    if method == "slic":
        import numpy as np
        from PIL import Image
        from skimage.segmentation import slic

        with Image.open(image) as source:
            image_array = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
        labels = slic(image_array, n_segments=superpixels, compactness=10, start_label=0)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels.astype(np.uint16)).save(label)
        return

    if method in {"sicle_comp", "sicle_irreg"}:
        executable = "RunSICLECOMP" if method == "sicle_comp" else "RunSICLEIRREG"
        command = [str(SICLE_BIN / executable), "--img", str(image),
                   "--out", str(label), "--nf", str(superpixels), "--n0", str(superpixels * 2)]
        run(command)
        return

    if method == "LSC":
        run([str(LSC_BIN), "--img", str(image), "--out", str(label),
             "--num-sp", str(superpixels)])
        return

    if method == "SEEDS":
        import cv2
        import numpy as np

        image_array = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if image_array is None:
            raise RuntimeError(f"Não foi possível abrir a imagem: {image}")
        height, width = image_array.shape[:2]
        lab_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            width, height, 3, superpixels, 4, 2, 5, False
        )
        seeds.iterate(lab_image, 10)
        if not cv2.imwrite(str(label), seeds.getLabels().astype(np.uint16)):
            raise RuntimeError(f"Não foi possível salvar os labels: {label}")
        return

    if method == "SH":
        import numpy as np
        from PIL import Image

        if str(SH_PYBUILD) not in sys.path:
            sys.path.insert(0, str(SH_PYBUILD))
        boruvka_superpixel = import_module("boruvka_superpixel")

        with Image.open(image) as source:
            image_array = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
        border_strength = np.zeros(image_array.shape[:2], dtype=image_array.dtype)
        segmenter = boruvka_superpixel.BoruvkaSuperpixel()
        segmenter.build_2d(image_array, border_strength)
        labels = segmenter.label(superpixels)
        Image.fromarray(np.asarray(labels).astype(np.uint16)).save(label)
        return

    raise ValueError(f"Método de segmentação desconhecido: {method}")

def meanLabel(image: Path, label: Path, output_base: Path) -> Path:
    output_png = output_base.with_suffix(".png")
    run([str(SICLE_BIN / "RunMeanLabel"), "--img", str(image), "--labels", str(label), "--out", str(output_png)])
    if not output_png.is_file():
        raise RuntimeError(f"Mean label não gerado: {output_png}")
    return output_png

def vectorize(method: str, image: Path, label: Path, output_base: Path) -> Path:
    run([str(SICLE_BIN / "RunVILLI"), "--img", str(image),
         "--labels", str(label), "--out", str(output_base),
            "--method", "sicle" if method in {"sicle_comp", "sicle_irreg"} else "slic"])
    svg = output_base.with_suffix(".svg")
    if not svg.is_file():
        raise RuntimeError(f"SVG não gerado: {svg}")
    return svg


def resolve_ground_truth_path(image: Path, gt_dir: Path | None) -> Path | None:
    if gt_dir is None:
        return None
    matches = sorted(gt_dir.glob(f"{image.stem}.*"))
    if matches:
        return matches[0]
    if (gt_dir / image.name).exists():
        return gt_dir / image.name
    return None


def metric_values_superpixel(label: Path, gt_path: Path | None) -> tuple[str, str]:
    if gt_path is None:
        return "", ""

    command = [
        str(SICLE_BIN / "RunSuperspelMetrics"),
        "--labels", str(label),
        "--gt", str(gt_path),
        "--br", "--ue", "--csv",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("metricas BR/UE falharam:\n" + completed.stdout[-2000:])

    raw = completed.stdout.strip()
    if not raw:
        return "", ""

    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) < 3:
        return "", ""
    return f"{float(values[1]):.8f}", f"{float(values[2]):.8f}"


def metric_values_svg(original: Path, svg: Path, modules, lpips_model):
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



def metric_values_mean_label(original: Path, mean_label: Path, modules, lpips_model):
    cairosvg, _, np, torch, Image, psnr_fn, ssim_fn = modules

    rendered_bytes = mean_label

    with mean_label.open("rb") as buffer:
        rendered_bytes = buffer.read()

    with Image.open(original) as source:
        source = source.convert("RGB")
        width, height = source.size
        reference = np.array(source, dtype=np.uint8, copy=True)

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


def read_rows(path: Path) -> list[dict[str,object]]: 
    if not path.exists():
        return []

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):   
            normalized: dict[str, object] = {
                str(key): value for key, value in row.items()
            }
            if normalized.get("image") != "__mean__":
                rows.append(normalized)
    return rows

def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        fieldnames = list(fieldnames)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        return

    successful = [row for row in rows if row["status"] == "ok"]
    saved_key = "saved_svg" if fieldnames == FIELDS_SVG else "saved_mean_img"
    mean = {"image": "__mean__", "method": rows[0]["method"], "superpixels": rows[0]["superpixels"],
            "status": "summary", saved_key: "", "error": ""}
    for field in ("processing_time_s", "mse", "psnr", "ssim", "lpips", "br", "ue"):
        mean[field] = sum(parse_float(row[field]) for row in successful if row.get(field) not in ("", None)) / len([row for row in successful if row.get(field) not in ("", None)]) if any(row.get(field) not in ("", None) for row in successful) else ""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean)


def vectorization_configuration(method: str, superpixels: int, images: list[Path], save_limit: int, modules, lpips_model, gt_dir: Path | None = None) -> None:
    csv_path = RESULTS_DIR_SVG / f"{method}_{superpixels}.csv"
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
        label = LABEL_ROOT / method / str(superpixels) / f"{image.stem}.pgm"
        output_base = temp_dir / image.stem
        temporary_svg = output_base.with_suffix(".svg")
        start = time.perf_counter()
        row = {"image": image.name, "method": method, "superpixels": superpixels,
               "status": "error", "saved_svg": "false", "processing_time_s": "",
               "mse": "", "psnr": "", "ssim": "", "lpips": "", "br": "", "ue": "", "error": ""}
        try:
            segment(method, image, label, superpixels)
            gt_path = resolve_ground_truth_path(image, gt_dir)
            br_score, ue_score = metric_values_superpixel(label, gt_path)
            svg = vectorize(method, image, label, output_base)
            elapsed = time.perf_counter() - start
            mse, psnr, ssim, lpips_score = metric_values_svg(image, svg, modules, lpips_model)
            row.update(status="ok", processing_time_s=f"{elapsed:.6f}", mse=f"{mse:.8f}",
                       psnr=f"{psnr:.8f}", ssim=f"{ssim:.8f}", lpips=f"{lpips_score:.8f}",
                       br=br_score, ue=ue_score)
            if saved < save_limit:
                destination = output_dir / f"{image.stem}.svg"
                shutil.move(str(svg), destination)
                row["saved_svg"] = "true"
                saved += 1
        except Exception as exc:
            row["error"] = str(exc).replace("\n", " ")[:2000]
        finally:
            temporary_svg.unlink(missing_ok=True)
            output_base.with_suffix(".gri").unlink(missing_ok=True)

        rows: list[dict[str, object]] = [previous for previous in rows if previous["image"] != image.name]
        rows.append(row)
        write_rows(csv_path, rows, FIELDS_SVG)  # checkpoint: safe to resume after interruption

    shutil.rmtree(temp_dir, ignore_errors=True)

def mean_label_configuration(method: str, superpixels: int, images: list[Path], save_limit: int, modules, lpips_model, gt_dir: Path | None = None) -> None:
    csv_path = RESULTS_DIR_MEAN / f"{method}_{superpixels}.csv"
    rows = read_rows(csv_path)
    completed = {row["image"] for row in rows if row["status"] == "ok"}
    saved = sum(row["saved_mean_img"] == "true" for row in rows if row["status"] == "ok")
    output_dir = MEAN_ROOT / method / str(superpixels)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = TEMP_ROOT / method / str(superpixels)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(images, 1):
        if image.name in completed:
            continue
        print(f"[{method}/{superpixels}] {index}/{len(images)} {image.name}", flush=True)
        label = LABEL_ROOT / method / str(superpixels) / f"{image.stem}.pgm"
        output_base = temp_dir / image.stem
        temporary_output = output_base.with_suffix(".png")
        start = time.perf_counter()
        row = {"image": image.name, "method": method, "superpixels": superpixels,
               "status": "error", "saved_mean_img": "false", "processing_time_s": "",
               "mse": "", "psnr": "", "ssim": "", "lpips": "", "br": "", "ue": "", "error": ""}
        try:
            segment(method, image, label, superpixels)
            gt_path = resolve_ground_truth_path(image, gt_dir)
            br_score, ue_score = metric_values_superpixel(label, gt_path)
            mean_label = meanLabel(image, label, output_base)
            elapsed = time.perf_counter() - start
            mse, psnr, ssim, lpips_score = metric_values_mean_label(image, mean_label, modules, lpips_model)
            row.update(status="ok", processing_time_s=f"{elapsed:.6f}", mse=f"{mse:.8f}",
                       psnr=f"{psnr:.8f}", ssim=f"{ssim:.8f}", lpips=f"{lpips_score:.8f}",
                       br=br_score, ue=ue_score)
            if saved < save_limit:
                destination = output_dir / f"{image.stem}.png"
                shutil.move(str(mean_label), destination)
                row["saved_mean_img"] = "true"
                saved += 1
        except Exception as exc:
            row["error"] = str(exc).replace("\n", " ")[:2000]
        finally:
            temporary_output.unlink(missing_ok=True)

        rows: list[dict[str, object]] = [previous for previous in rows if previous["image"] != image.name]
        rows.append(row)
        write_rows(csv_path, rows, FIELDS_MEAN)  # checkpoint: safe to resume after interruption



def main() -> None:
    description = (
        "Benchmark de segmentações superpixel para os métodos SLIC, SICLE e SICLE irregular.\n\n"
        "O script processa imagens em assets/INPUT, gera rótulos e compara os resultados em dois fluxos: \n"
        "  1) SVG: usa RunVILLI para gerar uma representação vetorial e mede MSE, PSNR, SSIM e LPIPS;\n"
        "  2) Mean: usa RunMeanLabel para gerar a imagem média da região e mede MSE, PSNR, SSIM e LPIPS.\n\n"
        "Além disso, quando um diretório de ground-truth é informado, também são calculadas as métricas de\n"
        "superpixel BR (Boundary Recall) e UE (Under-segmentation Error) via RunSuperspelMetrics."
    )
    epilog = (
        "Exemplos de uso:\n"
        "  python3 ./metrics/metrics.py --metric-mode svg\n"
        "  python3 ./metrics/metrics.py --metric-mode mean\n"
        "  python3 ./metrics/metrics.py --metric-mode both\n"
        "  python3 ./metrics/metrics.py --metric-mode both --gt-dir assets/LABEL --limit 5\n"
        "  python3 ./metrics/metrics.py --methods slic --superpixels 500 2000 --metric-mode mean"
    )
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS,
                        help="métodos a serem avaliados: slic, sicle_comp, sicle_irreg")
    parser.add_argument("--superpixels", nargs="+", type=int, choices=SUPERPIXELS,
                        default=SUPERPIXELS,
                        help="número de superpixels para testar: 500, 2000, 4000")
    parser.add_argument("--limit", type=int, default=None,
                        help="processa somente as primeiras N imagens (teste)")
    parser.add_argument("--save-limit", type=int, default=30,
                        help="limita quantas imagens salvas em disco por métrica (default: 30)")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="diretório de ground-truth para calculo de BR e UE; o nome do arquivo deve coincidir com a imagem de entrada")
    parser.add_argument("--metric-mode", choices=METRIC_MODES, default="both",
                        help="modo de benchmark: svg, mean ou both")
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
            if args.metric_mode in {"svg", "both"}:
                vectorization_configuration(method, superpixels, images, args.save_limit, modules, lpips_model, args.gt_dir)
            if args.metric_mode in {"mean", "both"}:
                mean_label_configuration(method, superpixels, images, args.save_limit, modules, lpips_model, args.gt_dir)


if __name__ == "__main__":
    main()
