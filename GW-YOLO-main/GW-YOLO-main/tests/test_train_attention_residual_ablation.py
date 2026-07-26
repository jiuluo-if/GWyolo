from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.train_attention_residual_ablation import (
    DEFAULT_SEEDS,
    DEFAULT_VARIANTS,
    build_jobs,
    locked_train_args,
    seed_list,
    transfer_matching_module_state,
    validate_inputs,
    variant_list,
    write_manifest,
)


class AttentionResidualTrainingTests(unittest.TestCase):
    def test_default_matrix_has_three_variants_and_three_seeds(self) -> None:
        jobs = build_jobs(DEFAULT_VARIANTS, DEFAULT_SEEDS, Path("runs/test"))
        self.assertEqual(len(jobs), 9)
        self.assertEqual(len({job.run_name for job in jobs}), 9)
        self.assertEqual(jobs[0].run_name, "baseline-seed0")
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


if __name__ == "__main__":
    unittest.main()
