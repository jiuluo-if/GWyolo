"""执行锁定变量、多随机种子的 Attention Residual 分割正式训练。

实验只改变模型拓扑：

- baseline：无额外 P3/P4 细化
- p4：仅在 P4 使用残差注意力
- p3p4：在 P3 和 P4 使用残差注意力

所有训练超参数、增强、数据路径和初始化权重均共享，使结果能够支持架构归因。
训练前记录数据、模型配置和预训练权重的 SHA-256；训练中逐任务写入状态，支持在
``last.pt`` 存在时使用 ``--resume`` 安全恢复。使用 ``--dry-run`` 可在不导入
Ultralytics 或启动 GPU 的情况下审计完整任务矩阵。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_VARIANTS = ("baseline", "p4", "p3p4")
DEFAULT_SEEDS = (0, 1, 2)
EXPECTED_HEAD_TRANSFER = (362, 376)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
LABEL_SUFFIXES = {".txt"}
MANIFEST_NAME = "experiment_manifest.json"
STATUS_NAME = "training_status.json"
COMPLETION_NAME = "training_complete.json"
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
    """按种子交错结构，避免同一结构连续长跑造成明显的时间顺序偏置。"""
    return [
        TrainingJob(
            variant=variant,
            seed=seed,
            config=VARIANT_CONFIGS[variant],
            run_name=f"{variant}-seed{seed}",
            output_dir=project / f"{variant}-seed{seed}",
        )
        for seed in seeds
        for variant in variants
    ]


def normalize_project_path(project: Path) -> Path:
    """使用绝对项目路径，避免 Ultralytics 再次前缀 runs_dir。"""
    return project.expanduser().resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _path_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def _resolve_dataset_entry(dataset_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def dataset_files(data: Path) -> tuple[Path, list[Path], list[Path]]:
    """解析数据配置并返回数据根目录、图像与标签文件。

    缓存和训练产物不进入快照，因此 Ultralytics 更新 ``labels.cache`` 不会破坏续训。
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("生成数据快照需要 PyYAML，请先安装 requirements.txt") from exc

    data = data.expanduser().resolve()
    payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"数据配置必须是 YAML 映射：{data}")

    root_value = payload.get("path", data.parent)
    dataset_root = Path(root_value).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = data.parent / dataset_root
    dataset_root = dataset_root.resolve()

    entries: list[Path] = []
    for key in ("train", "val", "labels"):
        entries.extend(
            _resolve_dataset_entry(dataset_root, value)
            for value in _path_values(payload.get(key))
        )
    if not entries:
        raise ValueError(f"数据配置未提供 train/val/labels 路径：{data}")

    missing = [entry for entry in entries if not entry.exists()]
    if missing:
        raise FileNotFoundError(
            "数据配置引用的路径不存在：" + ", ".join(map(str, missing))
        )

    files: set[Path] = set()
    for entry in entries:
        if entry.is_file():
            files.add(entry)
        else:
            files.update(path for path in entry.rglob("*") if path.is_file())
    images = sorted(path for path in files if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(path for path in files if path.suffix.lower() in LABEL_SUFFIXES)
    if not images or not labels:
        raise ValueError(
            f"数据快照必须同时包含图像和标签：images={len(images)}, labels={len(labels)}"
        )
    return dataset_root, images, labels


def dataset_snapshot(data: Path) -> dict[str, object]:
    """对固定训练/验证数据做内容级快照。"""
    data = data.expanduser().resolve()
    dataset_root, images, labels = dataset_files(data)
    digest = hashlib.sha256()
    total_bytes = 0
    for path in [*images, *labels]:
        try:
            stable_name = path.relative_to(dataset_root).as_posix()
        except ValueError:
            stable_name = path.as_posix()
        file_digest = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        digest.update(stable_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "data_config": str(data),
        "data_config_sha256": sha256_file(data),
        "dataset_root": str(dataset_root),
        "dataset_sha256": digest.hexdigest(),
        "image_count": len(images),
        "label_count": len(labels),
        "total_bytes": total_bytes,
    }


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
    """返回每种拓扑共同遵守的训练契约。"""
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
    """在模块间迁移局部名称相同且形状匹配的张量。

    自定义细化模块会将最终 Segment26 层移动到不同的外层索引。即使头部多数张量
    仍兼容，Ultralytics 的常规整模型加载也无法通过完整 state-dict 键匹配该头部。
    该局部重映射为每种拓扑提供相同的兼容预训练头部初始化，同时不触碰类别特定张量
    和新增注意力模块。
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
    allow_existing: bool = False,
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
    existing = [
        job.output_dir
        for job in jobs
        if job.output_dir.exists() and not allow_existing
    ]
    if existing:
        raise FileExistsError(
            "refusing to reuse existing run directories: "
            + ", ".join(map(str, existing))
        )


def plan_payload(
    jobs: Sequence[TrainingJob],
    args: argparse.Namespace,
    snapshot: dict[str, object],
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
        "protocol": "locked_attention_residual_ablation_v2",
        "purpose": "固定训练集、共享控制变量的三结构三随机种子正式训练",
        "selection_rule": (
            "仅用固定验证集校准和选模；GW5 按事件级规则盲评且不得调参；"
            "时间隔离纯负集报告 FP、false alarms/day；三种子报告均值和标准差"
        ),
        "gw5_event_rule": (
            "排除质量字段为 -- 的事件；H1/L1/V1 任一图像检出 class-0 chirp "
            "即召回该事件"
        ),
        "initialization": (
            "先加载整模型可匹配权重，再按局部键名和形状迁移兼容 Segment26 头部张量"
        ),
        "expected_head_transfer": {
            "transferred_tensors": EXPECTED_HEAD_TRANSFER[0],
            "target_tensors": EXPECTED_HEAD_TRANSFER[1],
        },
        "software": {
            "python": platform.python_version(),
            "ultralytics": package_version("ultralytics"),
            "torch": package_version("torch"),
        },
        "fingerprints": {
            "dataset": snapshot,
            "pretrained": {
                "path": str(args.pretrained.expanduser().resolve()),
                "sha256": sha256_file(args.pretrained),
            },
            "model_configs": {
                variant: {
                    "path": str(VARIANT_CONFIGS[variant].resolve()),
                    "sha256": sha256_file(VARIANT_CONFIGS[variant]),
                }
                for variant in args.variants
            },
        },
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
    path = project / MANIFEST_NAME
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment manifest: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def load_or_create_manifest(
    project: Path,
    payload: dict[str, object],
    *,
    resume: bool,
) -> Path:
    path = project / MANIFEST_NAME
    if not resume:
        return write_manifest(project, payload)
    if not path.is_file():
        raise FileNotFoundError(f"--resume 需要已有实验清单：{path}")
    existing = read_json(path)
    if existing != payload:
        raise ValueError(
            "当前参数或数据指纹与已有实验清单不一致，拒绝续训。"
            "请使用原始参数，或为新实验指定新的 --project。"
        )
    return path


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def initial_status(jobs: Sequence[TrainingJob]) -> dict[str, object]:
    return {
        "protocol": "locked_attention_residual_ablation_v2",
        "updated_at_utc": utc_now(),
        "jobs": {
            job.run_name: {
                "variant": job.variant,
                "seed": job.seed,
                "state": "pending",
                "output_dir": str(job.output_dir),
            }
            for job in jobs
        },
    }


def load_or_create_status(
    project: Path,
    jobs: Sequence[TrainingJob],
    *,
    resume: bool,
) -> tuple[Path, dict[str, object]]:
    path = project / STATUS_NAME
    if path.is_file():
        if not resume:
            raise FileExistsError(f"拒绝覆盖训练状态：{path}")
        payload = read_json(path)
    else:
        payload = initial_status(jobs)
        atomic_write_json(path, payload)
    expected_names = {job.run_name for job in jobs}
    status_jobs = payload.get("jobs")
    if not isinstance(status_jobs, dict) or set(status_jobs) != expected_names:
        raise ValueError(f"训练状态任务集合与当前计划不一致：{path}")
    return path, payload


def update_job_status(
    status_path: Path,
    status: dict[str, object],
    run_name: str,
    state: str,
    **details: object,
) -> None:
    jobs = status["jobs"]
    assert isinstance(jobs, dict)
    job_status = jobs[run_name]
    assert isinstance(job_status, dict)
    job_status.update({"state": state, **details})
    status["updated_at_utc"] = utc_now()
    atomic_write_json(status_path, status)


def completed_job(job: TrainingJob) -> bool:
    completion = job.output_dir / COMPLETION_NAME
    best = job.output_dir / "weights" / "best.pt"
    return completion.is_file() and best.is_file()


def resumable_checkpoint(job: TrainingJob) -> Path | None:
    checkpoint = job.output_dir / "weights" / "last.pt"
    return checkpoint if checkpoint.is_file() else None


def validate_resume_state(jobs: Iterable[TrainingJob], *, resume: bool) -> None:
    if not resume:
        return
    broken = [
        job.output_dir
        for job in jobs
        if job.output_dir.exists()
        and not completed_job(job)
        and resumable_checkpoint(job) is None
    ]
    if broken:
        raise FileNotFoundError(
            "以下已有任务既无完成标记也无 last.pt，无法安全续训："
            + ", ".join(map(str, broken))
        )


def verify_head_transfer(transferred: int, target_tensors: int) -> None:
    actual = (transferred, target_tensors)
    if actual != EXPECTED_HEAD_TRANSFER:
        raise RuntimeError(
            "预训练分割头迁移数量漂移："
            f"实际 {transferred}/{target_tensors}，"
            f"预期 {EXPECTED_HEAD_TRANSFER[0]}/{EXPECTED_HEAD_TRANSFER[1]}。"
            "请检查 Ultralytics 版本、模型配置和预训练权重。"
        )


def verify_save_dir(actual: Path | str, expected: Path) -> Path:
    """确保 Ultralytics 的真实输出没有偏离实验清单。"""
    actual_path = Path(actual).resolve()
    expected_path = expected.resolve()
    if actual_path != expected_path:
        raise RuntimeError(
            f"Ultralytics 输出目录漂移：实际 {actual_path}，预期 {expected_path}"
        )
    return actual_path


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
        default=Path("runs/segment/attention_residual_ablation_formal_v1"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="校验已有清单后跳过完成任务，并从未完成任务的 last.pt 恢复",
    )
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

    args.project = normalize_project_path(args.project)
    args.data = args.data.expanduser().resolve()
    args.pretrained = args.pretrained.expanduser().resolve()
    jobs = build_jobs(args.variants, args.seeds, args.project)
    validate_inputs(
        jobs,
        args.data,
        args.pretrained,
        require_runtime_inputs=True,
        allow_existing=args.resume,
    )
    snapshot = dataset_snapshot(args.data)
    payload = plan_payload(jobs, args, snapshot)
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    manifest_path = load_or_create_manifest(
        args.project,
        payload,
        resume=args.resume,
    )
    validate_resume_state(jobs, resume=args.resume)
    status_path, status = load_or_create_status(
        args.project,
        jobs,
        resume=args.resume,
    )
    print(f"Experiment manifest: {manifest_path}")
    print(f"Training status: {status_path}")

    from ultralytics import YOLO

    pretrained_reference = YOLO(args.pretrained)
    pretrained_head = pretrained_reference.model.model[-1]
    for index, job in enumerate(jobs, start=1):
        if completed_job(job):
            print(f"[{index}/{len(jobs)}] Skip completed {job.run_name}")
            update_job_status(
                status_path,
                status,
                job.run_name,
                "completed",
                skipped_on_resume=True,
            )
            continue
        print(
            f"[{index}/{len(jobs)}] variant={job.variant} seed={job.seed} "
            f"config={job.config}"
        )
        checkpoint = resumable_checkpoint(job) if args.resume else None
        started_at = utc_now()
        update_job_status(
            status_path,
            status,
            job.run_name,
            "running",
            started_at_utc=started_at,
            resume_checkpoint=str(checkpoint) if checkpoint else None,
        )
        try:
            if checkpoint:
                print(f"Resume from checkpoint: {checkpoint}")
                model = YOLO(checkpoint)
                result = model.train(resume=True)
                transferred = None
                target_tensors = None
            else:
                model = YOLO(job.config)
                model.load(args.pretrained)
                transferred, target_tensors = transfer_matching_module_state(
                    pretrained_head,
                    model.model.model[-1],
                )
                verify_head_transfer(transferred, target_tensors)
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
            save_dir = verify_save_dir(result.save_dir, job.output_dir)
            completion_payload = {
                "run_name": job.run_name,
                "variant": job.variant,
                "seed": job.seed,
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "save_dir": str(save_dir),
                "resumed_from": str(checkpoint) if checkpoint else None,
                "head_transfer": (
                    {
                        "transferred_tensors": transferred,
                        "target_tensors": target_tensors,
                    }
                    if transferred is not None
                    else None
                ),
            }
            atomic_write_json(
                save_dir / COMPLETION_NAME,
                completion_payload,
            )
            update_job_status(
                status_path,
                status,
                job.run_name,
                "completed",
                completed_at_utc=completion_payload["completed_at_utc"],
                save_dir=str(save_dir),
            )
            print(f"Completed {job.run_name}: {save_dir}")
        except Exception as exc:
            update_job_status(
                status_path,
                status,
                job.run_name,
                "failed",
                failed_at_utc=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            if "result" in locals():
                del result
            if "model" in locals():
                del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
