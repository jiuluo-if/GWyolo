"""Run a locked multi-seed Attention Residual segmentation ablation.

The experiment varies only the model topology:

- baseline: no extra P3/P4 refinement
- p4: residual attention on P4 only
- p3p4: residual attention on P3 and P4

All training hyperparameters, augmentations, data paths and initialization
weights are shared so the resulting runs can support an architecture claim.
Use ``--dry-run`` to audit the complete job matrix without importing
Ultralytics or starting GPU work.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_VARIANTS = ("baseline", "p4", "p3p4")
DEFAULT_SEEDS = (0, 1, 2)
VARIANT_CONFIGS = {
    "baseline": Path("configs/yolo26m-chirp-baseline-seg.yaml"),
    "p4": Path("configs/yolo26m-chirp-p4-attn-seg.yaml"),
    "p3p4": Path("configs/yolo26m-chirp-attn-seg.yaml"),
}


@dataclass(frozen=True)
class TrainingJob:
    variant: str
    seed: int
    config: Path
    run_name: str
    output_dir: Path


def comma_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return items


def seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in comma_list(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def variant_list(value: str) -> tuple[str, ...]:
    variants = comma_list(value)
    unknown = sorted(set(variants) - set(VARIANT_CONFIGS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variants: {', '.join(unknown)}; "
            f"choose from {', '.join(VARIANT_CONFIGS)}"
        )
    if len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError("variants must be unique")
    return variants


def build_jobs(
    variants: Sequence[str],
    seeds: Sequence[int],
    project: Path,
) -> list[TrainingJob]:
    return [
        TrainingJob(
            variant=variant,
            seed=seed,
            config=VARIANT_CONFIGS[variant],
            run_name=f"{variant}-seed{seed}",
            output_dir=project / f"{variant}-seed{seed}",
        )
        for variant in variants
        for seed in seeds
    ]


def locked_train_args(
    *,
    data: Path,
    project: Path,
    run_name: str,
    seed: int,
    device: str,
    workers: int,
    epochs: int,
    batch: int,
) -> dict[str, object]:
    """Return the common training contract shared by every topology."""
    return {
        "data": str(data),
        "epochs": epochs,
        "imgsz": 640,
        "batch": batch,
        "workers": workers,
        "device": device,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "lrf": 0.01,
        "cos_lr": True,
        "warmup_epochs": 5.0,
        "weight_decay": 0.0005,
        "cls": 1.5,
        "box": 8.0,
        "mask_ratio": 2,
        "overlap_mask": True,
        "amp": True,
        "mosaic": 0.35,
        "close_mosaic": 20,
        "mixup": 0.05,
        "translate": 0.08,
        "scale": 0.20,
        "erasing": 0.15,
        "degrees": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "patience": 50,
        "save": True,
        "save_period": 10,
        "seed": seed,
        "deterministic": True,
        "project": str(project),
        "name": run_name,
        "exist_ok": False,
    }


def transfer_matching_module_state(source_module, target_module) -> tuple[int, int]:
    """Transfer locally named tensors with matching shapes between modules.

    Custom refinement blocks move the final Segment26 layer to a different
    outer index. Ultralytics' normal full-model load cannot match that head by
    its full state-dict key, even though most tensors inside the head remain
    compatible. This local remap gives every topology the same compatible
    pretrained head initialization while leaving class-specific tensors and
    new attention blocks untouched.
    """
    source_state = source_module.state_dict()
    target_state = target_module.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    target_module.load_state_dict(compatible, strict=False)
    return len(compatible), len(target_state)


def validate_inputs(
    jobs: Sequence[TrainingJob],
    data: Path,
    pretrained: Path,
    *,
    require_runtime_inputs: bool,
) -> None:
    missing_configs = sorted({job.config for job in jobs if not job.config.is_file()})
    if missing_configs:
        raise FileNotFoundError(
            "missing model configs: " + ", ".join(map(str, missing_configs))
        )
    if require_runtime_inputs and not data.is_file():
        raise FileNotFoundError(f"dataset config not found: {data}")
    if require_runtime_inputs and not pretrained.is_file():
        raise FileNotFoundError(f"pretrained weights not found: {pretrained}")
    existing = [job.output_dir for job in jobs if job.output_dir.exists()]
    if existing:
        raise FileExistsError(
            "refusing to reuse existing run directories: "
            + ", ".join(map(str, existing))
        )


def plan_payload(
    jobs: Sequence[TrainingJob],
    args: argparse.Namespace,
) -> dict[str, object]:
    common = locked_train_args(
        data=args.data,
        project=args.project,
        run_name="<variant>-seed<seed>",
        seed=-1,
        device=args.device,
        workers=args.workers,
        epochs=args.epochs,
        batch=args.batch,
    )
    common.pop("name")
    common.pop("seed")
    return {
        "protocol": "locked_attention_residual_ablation_v1",
        "selection_rule": (
            "compare validation metrics and GW5 event recall with identical "
            "data, hyperparameters and evaluation thresholds; report mean/std"
        ),
        "initialization": (
            "load all full-model key matches, then remap compatible Segment26 "
            "head tensors by local key and shape"
        ),
        "pretrained": str(args.pretrained),
        "common_train_args": common,
        "jobs": [
            {
                **asdict(job),
                "config": str(job.config),
                "output_dir": str(job.output_dir),
            }
            for job in jobs
        ],
    }


def write_manifest(project: Path, payload: dict[str, object]) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    path = project / "experiment_manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment manifest: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        type=variant_list,
        default=DEFAULT_VARIANTS,
        help="comma-separated subset of baseline,p4,p3p4",
    )
    parser.add_argument(
        "--seeds",
        type=seed_list,
        default=DEFAULT_SEEDS,
        help="comma-separated non-negative seeds",
    )
    parser.add_argument("--data", type=Path, default=Path("gw_data.yaml"))
    parser.add_argument("--pretrained", type=Path, default=Path("yolo26m-seg.pt"))
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/segment/attention_residual_ablation"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the locked matrix without importing Ultralytics",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch < 1:
        raise ValueError("batch must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")

    jobs = build_jobs(args.variants, args.seeds, args.project)
    validate_inputs(
        jobs,
        args.data,
        args.pretrained,
        require_runtime_inputs=not args.dry_run,
    )
    payload = plan_payload(jobs, args)
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    manifest_path = write_manifest(args.project, payload)
    print(f"Experiment manifest: {manifest_path}")

    from ultralytics import YOLO

    pretrained_reference = YOLO(args.pretrained)
    pretrained_head = pretrained_reference.model.model[-1]
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] variant={job.variant} seed={job.seed} "
            f"config={job.config}"
        )
        model = YOLO(job.config)
        model.load(args.pretrained)
        transferred, target_tensors = transfer_matching_module_state(
            pretrained_head,
            model.model.model[-1],
        )
        print(
            f"Head initialization: transferred {transferred}/{target_tensors} "
            "shape-compatible tensors"
        )
        result = model.train(
            **locked_train_args(
                data=args.data,
                project=args.project,
                run_name=job.run_name,
                seed=job.seed,
                device=args.device,
                workers=args.workers,
                epochs=args.epochs,
                batch=args.batch,
            )
        )
        print(f"Completed {job.run_name}: {result.save_dir}")
        del result, model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
