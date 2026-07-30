from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_attention_residual_ablation import (
    COMPLETION_NAME,
    DEFAULT_SEEDS,
    DEFAULT_VARIANTS,
    EXPECTED_HEAD_TRANSFER,
    atomic_write_json,
    build_jobs,
    completed_job,
    dataset_snapshot,
    initial_status,
    load_or_create_manifest,
    locked_train_args,
    normalize_project_path,
    resumable_checkpoint,
    seed_list,
    transfer_matching_module_state,
    validate_inputs,
    validate_resume_state,
    variant_list,
    verify_head_transfer,
    verify_save_dir,
    write_manifest,
)


class AttentionResidualTrainingTests(unittest.TestCase):
    def test_default_matrix_has_three_variants_and_three_seeds(self) -> None:
        jobs = build_jobs(DEFAULT_VARIANTS, DEFAULT_SEEDS, Path("runs/test"))
        self.assertEqual(len(jobs), 9)
        self.assertEqual(len({job.run_name for job in jobs}), 9)
        self.assertEqual(jobs[0].run_name, "baseline-seed0")
        self.assertEqual(jobs[1].run_name, "p4-seed0")
        self.assertEqual(jobs[3].run_name, "baseline-seed1")
        self.assertEqual(jobs[-1].run_name, "p3p4-seed2")

    def test_locked_arguments_preserve_chirp_direction(self) -> None:
        values = locked_train_args(
            data=Path("gw_data.yaml"),
            project=Path("runs/test"),
            run_name="p4-seed1",
            seed=1,
            device="0",
            workers=0,
            epochs=300,
            batch=2,
        )
        self.assertEqual(values["optimizer"], "AdamW")
        self.assertEqual(values["imgsz"], 640)
        self.assertEqual(values["degrees"], 0.0)
        self.assertEqual(values["flipud"], 0.0)
        self.assertEqual(values["fliplr"], 0.0)
        self.assertTrue(values["deterministic"])

    def test_project_path_is_absolute_before_ultralytics_receives_it(self) -> None:
        normalized = normalize_project_path(Path("runs/test"))
        self.assertTrue(normalized.is_absolute())
        self.assertTrue(str(normalized).endswith(str(Path("runs/test"))))

    def test_variant_and_seed_parsers_reject_invalid_values(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            variant_list("baseline,unknown")
        with self.assertRaises(argparse.ArgumentTypeError):
            seed_list("0,0")
        with self.assertRaises(argparse.ArgumentTypeError):
            seed_list("-1")

    def test_existing_output_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "runs"
            jobs = build_jobs(("baseline",), (0,), project)
            jobs[0].output_dir.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                validate_inputs(
                    jobs,
                    Path("unused.yaml"),
                    Path("unused.pt"),
                    require_runtime_inputs=False,
                )

    def test_head_transfer_uses_local_name_and_shape(self) -> None:
        class Value:
            def __init__(self, shape: tuple[int, ...]) -> None:
                self.shape = shape

        class Module:
            def __init__(self, state: dict[str, Value]) -> None:
                self._state = state
                self.loaded: dict[str, Value] = {}

            def state_dict(self) -> dict[str, Value]:
                return self._state

            def load_state_dict(
                self, state: dict[str, Value], strict: bool
            ) -> None:
                self.loaded = state
                self.strict = strict

        source = Module(
            {
                "shared.weight": Value((4, 4)),
                "class.weight": Value((80, 4)),
                "source_only": Value((1,)),
            }
        )
        target = Module(
            {
                "shared.weight": Value((4, 4)),
                "class.weight": Value((2, 4)),
                "target_only": Value((1,)),
            }
        )
        transferred, total = transfer_matching_module_state(source, target)
        self.assertEqual((transferred, total), (1, 3))
        self.assertEqual(set(target.loaded), {"shared.weight"})
        self.assertFalse(target.strict)

    def test_manifest_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "experiment"
            path = write_manifest(project, {"protocol": "test"})
            self.assertTrue(path.is_file())
            with self.assertRaises(FileExistsError):
                write_manifest(project, {"protocol": "replacement"})

    def test_resume_requires_identical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "experiment"
            expected = {"protocol": "v2", "jobs": ["baseline-seed0"]}
            write_manifest(project, expected)
            path = load_or_create_manifest(project, expected, resume=True)
            self.assertTrue(path.is_file())
            with self.assertRaises(ValueError):
                load_or_create_manifest(
                    project,
                    {"protocol": "changed"},
                    resume=True,
                )

    def test_resume_state_accepts_last_checkpoint_or_completed_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            jobs = build_jobs(("baseline", "p4"), (0,), project)
            last = jobs[0].output_dir / "weights" / "last.pt"
            last.parent.mkdir(parents=True)
            last.write_bytes(b"checkpoint")
            best = jobs[1].output_dir / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"checkpoint")
            atomic_write_json(
                jobs[1].output_dir / COMPLETION_NAME,
                {"completed": True},
            )
            validate_resume_state(jobs, resume=True)
            self.assertEqual(resumable_checkpoint(jobs[0]), last)
            self.assertTrue(completed_job(jobs[1]))

    def test_resume_state_rejects_unrecoverable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            jobs = build_jobs(("baseline",), (0,), Path(directory))
            jobs[0].output_dir.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                validate_resume_state(jobs, resume=True)

    def test_dataset_snapshot_ignores_cache_and_detects_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "labels").mkdir(parents=True)
            (root / "val" / "images").mkdir(parents=True)
            (root / "val" / "labels").mkdir(parents=True)
            (root / "train" / "images" / "a.png").write_bytes(b"image-a")
            (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5")
            (root / "val" / "images" / "b.png").write_bytes(b"image-b")
            (root / "val" / "labels" / "b.txt").write_text("0 0.5 0.5")
            (root / "val" / "labels.cache").write_bytes(b"ignored")
            data = root / "data.yaml"
            data.write_text(
                "\n".join(
                    [
                        f"path: {root.as_posix()}",
                        "train: train/images",
                        "labels: train/labels",
                        "val: val",
                    ]
                ),
                encoding="utf-8",
            )
            first = dataset_snapshot(data)
            (root / "val" / "labels.cache").write_bytes(b"changed cache")
            second = dataset_snapshot(data)
            self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
            self.assertEqual(first["image_count"], 2)
            self.assertEqual(first["label_count"], 2)

    def test_expected_head_transfer_is_enforced(self) -> None:
        verify_head_transfer(*EXPECTED_HEAD_TRANSFER)
        with self.assertRaises(RuntimeError):
            verify_head_transfer(EXPECTED_HEAD_TRANSFER[0] - 1, EXPECTED_HEAD_TRANSFER[1])

    def test_save_dir_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "expected"
            self.assertEqual(verify_save_dir(expected, expected), expected.resolve())
            with self.assertRaises(RuntimeError):
                verify_save_dir(Path(directory) / "nested", expected)

    def test_initial_status_contains_every_job(self) -> None:
        jobs = build_jobs(("baseline", "p4"), (0,), Path("runs/test"))
        status = initial_status(jobs)
        self.assertEqual(
            set(status["jobs"]),
            {"baseline-seed0", "p4-seed0"},
        )
        json.dumps(status)


if __name__ == "__main__":
    unittest.main()
