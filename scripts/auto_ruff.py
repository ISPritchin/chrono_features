# ruff: noqa: T201, BLE001, S607, S603, EXE002, INP001

import time
import subprocess
import argparse
from pathlib import Path


def run_ruff_fix(directory: str, *, verbose: bool = False) -> None:
    """Run ruff --fix on the specified directory.

    Args:
        directory: Directory to run ruff on
        verbose: Whether to print verbose output
    """
    if verbose:
        print(f"Running ruff --fix on {directory}...")

    try:
        result = subprocess.run(
            ["ruff", "check", "--fix", directory],
            capture_output=not verbose,
            text=True,
            check=False,
        )

        if verbose:
            if result.returncode == 0:
                print("Ruff fix completed successfully.")
            else:
                print(f"Ruff fix completed with issues: {result.stderr}")
    except Exception as e:
        print(f"Error running ruff: {e}")


def main() -> None:
    """Main function to run the auto-ruff script."""
    parser = argparse.ArgumentParser(description="Auto-run ruff --fix every 10 seconds")
    parser.add_argument(
        "--directory",
        "-d",
        default=str(Path.cwd()),
        help="Directory to run ruff on (default: current directory)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=60,
        help="Interval in seconds between ruff runs (default: 10)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    print(f"Auto-running ruff --fix on {args.directory} every {args.interval} seconds.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            run_ruff_fix(args.directory, args.verbose)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping auto-ruff.")


if __name__ == "__main__":
    main()
