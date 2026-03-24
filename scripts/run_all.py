from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(str(c) for c in cmd))
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.parent.resolve())
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
    subprocess.run(cmd, check=True, env=env)


def _append_subset_args(cmd: list[str], train_subset_ratio: float | None, train_subset_size: int | None) -> list[str]:
    if train_subset_ratio is not None:
        cmd.extend(["--train-subset-ratio", str(train_subset_ratio)])
    elif train_subset_size is not None:
        cmd.extend(["--train-subset-size", str(train_subset_size)])
    return cmd


def _append_resume_arg(cmd: list[str], resume: bool) -> list[str]:
    if resume:
        cmd.append("--resume")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full project pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("src/dataset"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", action="store_true", help="Resume training from saved train_state.pt files")
    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument("--train-subset-ratio", type=float, default=None)
    subset_group.add_argument("--train-subset-size", type=int, default=None)
    args = parser.parse_args()

    py = sys.executable

    cnn_out = args.out_root / "cnn_grid"
    eff_out = args.out_root / "efficientnet"
    fs_out = args.out_root / "fewshot"

    
    _run(
        _append_resume_arg(
            _append_subset_args(
                [py, "scripts/run_efficientnet.py", "--data-dir", str(args.data_dir), "--out-dir", str(eff_out)],
                args.train_subset_ratio,
                args.train_subset_size,
            ),
            args.resume,
        )
    )
    _run(
        _append_resume_arg(
            _append_subset_args(
                [py, "scripts/run_fewshot.py", "--data-dir", str(args.data_dir), "--out-dir", str(fs_out)],
                args.train_subset_ratio,
                args.train_subset_size,
            ),
            args.resume,
        )
    )
    _run(
        _append_resume_arg(
            _append_subset_args(
                [py, "scripts/run_reduced_data.py", "--data-dir", str(args.data_dir), "--out-dir", str(args.out_root / "reduced_data")],
                args.train_subset_ratio,
                args.train_subset_size,
            ),
            args.resume,
        )
    )
    
    _run(
        _append_resume_arg(
            _append_subset_args(
                [py, "scripts/run_cnn_grid.py", "--data-dir", str(args.data_dir), "--out-dir", str(cnn_out)],
                args.train_subset_ratio,
                args.train_subset_size,
            ),
            args.resume,
        )
    )

    with (cnn_out / "summary.json").open("r", encoding="utf-8") as f:
        cnn_summary = json.load(f)
    best_cnn_ckpt = cnn_summary["best"]["checkpoint"]

    _run(
        [
            py,
            "scripts/run_ensemble.py",
            "--data-dir",
            str(args.data_dir),
            "--cnn-ckpt",
            str(best_cnn_ckpt),
            "--effnet-ckpt",
            str(eff_out / "best.pt"),
            "--fewshot-ckpt",
            str(fs_out / "best.pt"),
            "--out-file",
            str(args.out_root / "ensemble" / "result.json"),
        ]
    )


if __name__ == "__main__":
    main()
