import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    print(f"[CWD] {cwd}\n")

    try:
        subprocess.check_call(cmd, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FASTA download + ESM embedding pipeline"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=os.path.abspath(os.path.join("..", "virtues")),
        help="Root directory of virtues project",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to markers CSV",
    )

    parser.add_argument(
        "--fastas-dir",
        type=Path,
        required=True,
        help="Output directory for FASTA files",
    )

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Output directory for embeddings",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for embeddings (cpu / cuda)",
    )

    parser.add_argument(
        "--model",
        default="esm2_t30_150M_UR50D",
        help="ESM model name",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root: Path = args.project_root.resolve()

    python_exec = sys.executable

    cmd_download = [
        python_exec,
        "-m",
        "utils.download_fastas",
        "--output_dir",
        str(args.fastas_dir),
        "--csv",
        str(args.csv),
    ]

    run_command(cmd_download, cwd=project_root)

    cmd_embeddings = [
        python_exec,
        "-m",
        "utils.compute_esm_embeddings",
        "--input_dir",
        str(args.fastas_dir),
        "--output_dir",
        str(args.embeddings_dir),
        "--device",
        args.device,
        "--model",
        args.model,
    ]

    run_command(cmd_embeddings, cwd=project_root)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
