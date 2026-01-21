#!/usr/bin/env python3
"""
scripts/make_small_benchmark_table.py

Creates a throughput results table for the SMALL model from Slurm array .out files.

Input directory (default):
  /home/scur2265/MLOps_2026/experiments/results/question 8/model changes/small_job

Outputs (default):
  assets/tabels/q8_small_throughput.md
  assets/tabels/q8_small_throughput.csv

Run from repo root (MLOps_2026):
  python scripts/make_small_benchmark_table.py

Or override paths:
  python scripts/make_small_benchmark_table.py \
    --in_dir "/home/scur2265/MLOps_2026/experiments/results/question 8/model changes/small_job" \
    --out_dir "assets/tabels"
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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
    error: Optional[str]
    file: str


_RE_FLOAT = r"[-+]?\d*\.\d+|\d+"


def _find_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return int(m.group(1)) if m else None


def _find_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def _find_str(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_out(path: Path) -> Row:
    txt = path.read_text(errors="replace")

    task_id = _find_int(r"^TaskID=(\d+)\b", txt)
    batch_size = _find_int(r"^Batch size\s*:\s*(\d+)\b", txt)

    model_params = _find_int(r"^Model params\s*:\s*(\d+)\b", txt)
    model_size_mb = _find_float(r"^Model size MB\s*:\s*(" + _RE_FLOAT + r")\b", txt)

    total_images = _find_int(r"^Total images\s*:\s*(\d+)\b", txt)
    elapsed_s = _find_float(r"^Elapsed \(s\)\s*:\s*(" + _RE_FLOAT + r")\b", txt)
    throughput = _find_float(
        r"^Throughput\s*:\s*(" + _RE_FLOAT + r")\s*images/sec\b", txt
    )

    gpu_name = _find_str(r"^GPU name\s*:\s*(.+)$", txt)

    # Errors (if any)
    error = _find_str(r"^ERROR\s*:\s*(.+)$", txt)
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
        error=error,
        file=str(path),
    )


def sort_rows(rows: List[Row]) -> List[Row]:
    return sorted(
        rows,
        key=lambda r: (
            r.batch_size if r.batch_size is not None else 10**9,
            r.task_id if r.task_id is not None else 10**9,
            r.file,
        ),
    )


def to_markdown(rows: List[Row], title: str) -> str:
    rows = sort_rows(rows)

    def fmt_i(x: Optional[int]) -> str:
        return str(x) if x is not None else "N/A"

    def fmt_f(x: Optional[float], nd: int) -> str:
        return f"{x:.{nd}f}" if x is not None else "N/A"

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("Generated from Slurm `.out` files.")
    lines.append("")
    lines.append(
        "| Batch size | Throughput (img/s) | Elapsed (s) | Total images | Error |"
    )
    lines.append(
        "|-----------:|-------------------:|------------:|------------:|:------|"
    )

    for r in rows:
        lines.append(
            f"| {fmt_i(r.batch_size)} | {fmt_f(r.throughput, 2)} | {fmt_f(r.elapsed_s, 4)} | {fmt_i(r.total_images)} | {r.error or ''} |"
        )

    model_params = next(
        (r.model_params for r in rows if r.model_params is not None), None
    )
    model_size_mb = next(
        (r.model_size_mb for r in rows if r.model_size_mb is not None), None
    )
    gpu_name = next((r.gpu_name for r in rows if r.gpu_name is not None), None)

    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(
        f"- Model params: {model_params if model_params is not None else 'N/A'}"
    )
    lines.append(
        f"- Model size (fp32 params only): {model_size_mb:.2f} MB"
        if model_size_mb is not None
        else "- Model size: N/A"
    )
    lines.append(f"- GPU: {gpu_name if gpu_name is not None else 'N/A'}")
    lines.append("")
    return "\n".join(lines)


def write_csv(rows: List[Row], out_path: Path) -> None:
    rows = sort_rows(rows)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "batch_size",
                "throughput_img_s",
                "elapsed_s",
                "total_images",
                "model_params",
                "model_size_mb",
                "gpu_name",
                "error",
                "file",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.batch_size,
                    r.throughput,
                    r.elapsed_s,
                    r.total_images,
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
        "--in_dir",
        default="/home/scur2265/MLOps_2026/experiments/results/question 8/model changes/small_job",
        help="Directory containing array_small_*.out files (can include spaces).",
    )
    ap.add_argument(
        "--pattern",
        default="array_small_*.out",
        help="Filename glob pattern inside --in_dir.",
    )
    ap.add_argument(
        "--out_dir",
        default="assets/tabels",
        help="Directory to write outputs (relative to repo root).",
    )
    ap.add_argument(
        "--title",
        default="Question 8 - SMALL model throughput benchmark (GPU)",
        help="Title for the Markdown output.",
    )
    ap.add_argument(
        "--out_name",
        default="q8_small_throughput",
        help="Base name for outputs (without extension).",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {in_dir}/{args.pattern}")

    rows = [parse_out(p) for p in files]

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
