# experiments/benchmark_throughput.py
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as T


@dataclass
class Result:
    mode: str
    batch_size: int
    num_workers: int
    steps: int
    total_images: int
    elapsed_s: float

    @property
    def img_per_s(self) -> float:
        return (
            self.total_images / self.elapsed_s if self.elapsed_s > 0 else float("inf")
        )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb_fp32(model: torch.nn.Module) -> float:
    # Approx: parameters only, float32 = 4 bytes
    return count_params(model) * 4 / (1024**2)


def build_small_mlp(dummy_input_shape):
    """
    Small baseline model (your current MLP).
    """
    c, h, w = dummy_input_shape
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(c * h * w, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 2),
    )


def build_big_model():
    """
    Much larger model for compute scaling analysis.
    ResNet50 is a standard "big" step up from a tiny MLP.
    """
    # ResNet50 expects 3-channel images; PCAM is RGB so that's fine.
    # Output classes = 2
    return models.resnet50(num_classes=2)


def resolve_pcam_paths(split: str, data_dir: Path) -> tuple[Path, Path]:
    direct_x = data_dir / f"{split}_x.h5"
    direct_y = data_dir / f"{split}_y.h5"
    if direct_x.exists() and direct_y.exists():
        return direct_x, direct_y

    candidates_x = sorted(data_dir.glob(f"*{split}*x*.h5"))
    candidates_y = sorted(data_dir.glob(f"*{split}*y*.h5"))

    if not candidates_x or not candidates_y:
        raise FileNotFoundError(
            f"Could not find PCAM H5 files for split='{split}' in '{data_dir}'. "
            f"Tried '{direct_x.name}', '{direct_y.name}' and glob patterns '*{split}*x*.h5' / '*{split}*y*.h5'."
        )
    return candidates_x[0], candidates_y[0]


def _is_cuda_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    # Common PyTorch CUDA OOM strings
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "cublas" in msg
        and "alloc" in msg
        or "cuda error: out of memory" in msg
    )


def _print_header(
    device: torch.device,
    model_name: str,
    model_params: int | None,
    model_mb: float | None,
    mode: str,
    split: str,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    steps: int,
    x_path: Path,
    y_path: Path,
    data_dir: Path,
) -> None:
    print("=== Throughput Benchmark ===")
    print(f"Device         : {device}")
    if device.type == "cuda":
        print(f"GPU name       : {torch.cuda.get_device_name(0)}")
    print(f"Model          : {model_name}")
    if model_params is not None and model_mb is not None:
        print(f"Model params   : {model_params}")
        print(f"Model size MB  : {model_mb:.2f} (fp32 params only)")
    print(f"Mode           : {mode}")
    print(f"Split          : {split}")
    print(f"Batch size     : {batch_size}")
    print(f"Num workers    : {num_workers}")
    print(f"Warmup steps   : {warmup_steps}")
    print(f"Measured steps : {steps}")
    print(f"x_path         : {x_path}")
    print(f"y_path         : {y_path}")
    print(f"data_dir       : {data_dir}")


def _print_vram(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        # Values are bytes
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_alloc = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        print(f"VRAM allocated : {allocated / (1024**2):.2f} MB")
        print(f"VRAM reserved  : {reserved / (1024**2):.2f} MB")
        print(f"VRAM max alloc : {max_alloc / (1024**2):.2f} MB")
        print(f"VRAM max reserv: {max_reserved / (1024**2):.2f} MB")
    except Exception:
        # Never let diagnostics crash the benchmark
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--steps", type=int, default=200, help="Measured steps (batches)."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=20, help="Warmup steps not counted."
    )
    parser.add_argument("--mode", choices=["dataloader", "forward"], default="forward")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model", choices=["small", "big"], default="small")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Useful for GPU transfers; harmless on CPU.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # Device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda but torch.cuda.is_available() is False."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from ml_core.data.pcam import PCAMDataset  # type: ignore

    project_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    x_path, y_path = resolve_pcam_paths(args.split, data_dir)

    # PCAM returns uint8 HWC; ToTensor -> float32 CHW in [0,1]
    transform = T.Compose([T.ToTensor()])

    dataset = PCAMDataset(
        x_path=str(x_path),
        y_path=str(y_path),
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if device.type == "cuda" else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        drop_last=True,
    )

    model = None
    model_params: int | None = None
    model_mb: float | None = None

    if args.mode == "forward":
        # Peek one batch to infer shape for the small MLP
        x0, _ = next(iter(loader))
        c, h, w = x0.shape[1], x0.shape[2], x0.shape[3]

        if args.model == "small":
            model = build_small_mlp((c, h, w))
        else:
            model = build_big_model()

        model_params = count_params(model)
        model_mb = model_size_mb_fp32(model)

        model = model.to(device)
        model.eval()

        # Initialize CUDA memory stats for reporting
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    # --- Warmup ---
    it = iter(loader)
    for _ in range(args.warmup_steps):
        x, _ = next(it)
        if args.mode == "forward":
            try:
                x = x.to(device, non_blocking=True)
                with torch.no_grad():
                    _ = model(x)  # type: ignore[misc]
            except RuntimeError as e:
                if device.type == "cuda" and _is_cuda_oom(e):
                    _print_header(
                        device=device,
                        model_name=args.model,
                        model_params=model_params,
                        model_mb=model_mb,
                        mode=args.mode,
                        split=args.split,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        warmup_steps=args.warmup_steps,
                        steps=args.steps,
                        x_path=x_path,
                        y_path=y_path,
                        data_dir=data_dir,
                    )
                    print("ERROR          : CUDA OOM during warmup")
                    _print_vram(device)
                    return
                raise

    # --- Measurement ---
    measured_steps = 0
    total_images = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    it = iter(loader)
    while measured_steps < args.steps:
        x, _ = next(it)
        bs = x.shape[0]

        if args.mode == "forward":
            try:
                x = x.to(device, non_blocking=True)
                with torch.no_grad():
                    _ = model(x)  # type: ignore[misc]
            except RuntimeError as e:
                if device.type == "cuda" and _is_cuda_oom(e):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    _print_header(
                        device=device,
                        model_name=args.model,
                        model_params=model_params,
                        model_mb=model_mb,
                        mode=args.mode,
                        split=args.split,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        warmup_steps=args.warmup_steps,
                        steps=args.steps,
                        x_path=x_path,
                        y_path=y_path,
                        data_dir=data_dir,
                    )
                    print(f"Total images   : {total_images}")
                    elapsed = time.perf_counter() - t0
                    print(f"Elapsed (s)    : {elapsed:.4f}")
                    print("Throughput     : N/A")
                    print("ERROR          : CUDA OOM during measurement")
                    _print_vram(device)
                    return
                raise

        total_images += bs
        measured_steps += 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    res = Result(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        total_images=total_images,
        elapsed_s=(t1 - t0),
    )

    # Normal successful output
    _print_header(
        device=device,
        model_name=args.model,
        model_params=model_params,
        model_mb=model_mb,
        mode=res.mode,
        split=args.split,
        batch_size=res.batch_size,
        num_workers=res.num_workers,
        warmup_steps=args.warmup_steps,
        steps=res.steps,
        x_path=x_path,
        y_path=y_path,
        data_dir=data_dir,
    )
    print(f"Total images   : {res.total_images}")
    print(f"Elapsed (s)    : {res.elapsed_s:.4f}")
    print(f"Throughput     : {res.img_per_s:.2f} images/sec")
    _print_vram(device)


if __name__ == "__main__":
    main()
