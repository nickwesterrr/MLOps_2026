#!/usr/bin/env python3
"""
scripts/make_big_benchmark_table.py

Parse Slurm array benchmark outputs for the BIG model and export a table
to assets/tabels as Markdown (+ CSV).

Expected input: one or more files matching a glob pattern, e.g.
  /home/scur2265/MLOps_2026/experiments/results/question 8/model changes/big_job/array_big_*.out

Run from repo root (MLOps_2026):
  python scripts/make_big_benchmark_table.py \
    --glob "/home/scur2265/MLOps_2026/experiments/results/question 8/model changes/big_job/array_big_*.out" \
    --out_dir "assets/tabels" \
    --out_name "q8_big_throughput"

Notes:
- Designed to work even if some tasks OOM and print "ERROR : CUDA OOM" (we mark throughput as N/A).
- Extracts: batch size, throughput, elapsed, total images, VRAM peaks, model size/params, GPU name, errors.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class Row:
    task_id: Optional[int]
    batch_size: Optional[int]
    throughput: Optional[float]
    elapsed_s: Optional[float]
    total_images: Optional[int]
    model_params: Optional[int]
    model_size_mb: Optional[float]
    gpu_name: Optional[str]
    vram_max_alloc_mb: Optional[float]
    vram_max_reserved_mb: Optional[float]
    error: Optional[str]
    file: str


_RE_FLOAT = r"[-+]?\d*\.\d+|\d+"


def _find_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return int(m.group(1))


def _find_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return float(m.group(1))


def _find_str(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def parse_out_file(path: Path) -> Row:
    txt = path.read_text(errors="replace")

    task_id = _find_int(r"^TaskID=(\d+)\b", txt)
    batch_size = _find_int(r"^Batch size\s*:\s*(\d+)\b", txt)

    model_params = _find_int(r"^Model params\s*:\s*(\d+)\b", txt)
    model_size_mb = _find_float(r"^Model size MB\s*:\s*(" + _RE_FLOAT + r")\b", txt)

    elapsed_s = _find_float(r"^Elapsed \(s\)\s*:\s*(" + _RE_FLOAT + r")\b", txt)
    total_images = _find_int(r"^Total images\s*:\s*(\d+)\b", txt)
    throughput = _find_float(r"^Throughput\s*:\s*(" + _RE_FLOAT + r")\s*images/sec\b", txt)

    gpu_name = _find_str(r"^GPU name\s*:\s*(.+)$", txt)

    vram_max_alloc_mb = _find_float(r"^VRAM max alloc\s*:\s*(" + _RE_FLOAT + r")\s*MB\b", txt)
    vram_max_reserved_mb = _find_float(r"^VRAM max reserv\s*:\s*(" + _RE_FLOAT + r")\s*MB\b", txt)

    # Detect explicit error line
    error = _find_str(r"^ERROR\s*:\s*(.+)$", txt)

    # Heuristics if throughput is missing but text hints at failure
    if error is None:
        if "out of memory" in txt.lower():
            error = "CUDA OOM (detected in stdout)"
        elif throughput is None and "=== Throughput Benchmark ===" in txt:
            error = "Missing throughput"

    return Row(
        task_id=task_id,
        batch_size=batch_size,
        throughput=throughput,
        elapsed_s=elapsed_s,
        total_images=total_images,
        model_params=model_params,
        model_size_mb=model_size_mb,
        gpu_name=gpu_name,
        vram_max_alloc_mb=vram_max_alloc_mb,
        vram_max_reserved_mb=vram_max_reserved_mb,
        error=error,
        file=str(path),
    )


def _sort_rows(rows: List[Row]) -> List[Row]:
    return sorted(
        rows,
        key=lambda r: (
            r.batch_size if r.batch_size is not None else 10**9,
            r.task_id if r.task_id is not None else 10**9,
            r.file,
        ),
    )


def to_markdown(rows: List[Row], title: str) -> str:
    rows_sorted = _sort_rows(rows)

    def fmt_f(x: Optional[float], nd: int = 2) -> str:
        return f"{x:.{nd}f}" if x is not None else "N/A"

    def fmt_i(x: Optional[int]) -> str:
        return f"{x}" if x is not None else "N/A"

    md: List[str] = []
    md.append(f"# {title}")
    md.append("")
    md.append("Generated from Slurm `.out` files.")
    md.append("")

    md.append(
        "| Batch size | Throughput (img/s) | Elapsed (s) | Total images | VRAM max alloc (MB) | VRAM max reserv (MB) | Error |"
    )
    md.append(
        "|-----------:|-------------------:|------------:|------------:|--------------------:|---------------------:|:------|"
    )

    for r in rows_sorted:
        md.append(
            f"| {fmt_i(r.batch_size)} | {fmt_f(r.throughput, 2)} | {fmt_f(r.elapsed_s, 4)} | {fmt_i(r.total_images)} | "
            f"{fmt_f(r.vram_max_alloc_mb, 2)} | {fmt_f(r.vram_max_reserved_mb, 2)} | {r.error or ''} |"
        )

    # Metadata footer
    model_params = next((r.model_params for r in rows_sorted if r.model_params is not None), None)
    model_size_mb = next((r.model_size_mb for r in rows_sorted if r.model_size_mb is not None), None)
    gpu_name = next((r.gpu_name for r in rows_sorted if r.gpu_name is not None), None)

    md.append("")
    md.append("## Metadata")
    md.append("")
    md.append(f"- Model params: {model_params if model_params is not None else 'N/A'}")
    md.append(
        f"- Model size (fp32 params only): {model_size_mb:.2f} MB" if model_size_mb is not None else "- Model size: N/A"
    )
    md.append(f"- GPU: {gpu_name if gpu_name is not None else 'N/A'}")
    md.append("")
    return "\n".join(md)


def write_csv(rows: List[Row], out_path: Path) -> None:
    rows_sorted = _sort_rows(rows)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "batch_size",
                "throughput_img_s",
                "elapsed_s",
                "total_images",
                "vram_max_alloc_mb",
                "vram_max_reserved_mb",
                "model_params",
                "model_size_mb",
                "gpu_name",
                "error",
                "file",
            ]
        )
        for r in rows_sorted:
            w.writerow(
                [
                    r.batch_size,
                    r.throughput,
                    r.elapsed_s,
                    r.total_images,
                    r.vram_max_alloc_mb,
                    r.vram_max_reserved_mb,
                    r.model_params,
                    r.model_size_mb,
                    r.gpu_name,
                    r.error,
                    r.file,
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        required=True,
        help="Glob pattern for BIG model .out files (quote it if it contains spaces).",
    )
    ap.add_argument(
        "--out_dir",
        default="assets/tabels",
        help="Directory to write the table files to (relative to repo root).",
    )
    ap.add_argument(
        "--title",
        default="Question 8 - BIG model throughput benchmark (GPU)",
        help="Title used in the Markdown output.",
    )
    ap.add_argument(
        "--out_name",
        default="q8_big_throughput",
        help="Base name for outputs (without extension).",
    )
    args = ap.parse_args()

    files = [Path(p) for p in glob.glob(args.glob)]
    files = sorted(files)

    if not files:
        raise SystemExit(f"No files matched glob: {args.glob}")

    rows = [parse_out_file(p) for p in files]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{args.out_name}.md"
    csv_path = out_dir / f"{args.out_name}.csv"

    md_path.write_text(to_markdown(rows, args.title))
    write_csv(rows, csv_path)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
