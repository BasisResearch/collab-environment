#!/usr/bin/env python
"""Convert all CSQ files in a directory to AVI/MP4 in parallel."""
import os
from pathlib import Path

import click
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich import print
from tqdm import tqdm

from collab_env.tracking.csq import csq_to_avi, detect_vmin_vmax


def run_conversion_job(
    input_path: Path,
    output_path: Path,
    vmin: float | None,
    vmax: float | None,
    max_mins: float | None,
) -> tuple[Path, str, str | None]:
    """Convert a single CSQ file."""
    try:
        if vmin is None or vmax is None:
            vmin, vmax = detect_vmin_vmax(str(input_path))
            if vmin is None or vmax is None:
                return output_path, "skipped", "could not detect vmin/vmax"
        csq_to_avi(
            str(input_path),
            vmin,
            vmax,
            max_mins=max_mins,
            output_path=str(output_path),
            show_progress=False,
            verbose=False,
        )
        return output_path, "done", None
    except Exception as e:
        return output_path, "error", str(e)


def default_max_workers() -> int:
    """Use all available CPU cores when max_workers not specified."""
    return os.cpu_count() or 1


@click.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--vmin",
    type=float,
    default=None,
    help="Min temperature for normalization. Auto-detected per file if omitted.",
)
@click.option(
    "--vmax",
    type=float,
    default=None,
    help="Max temperature for normalization. Auto-detected per file if omitted.",
)
@click.option(
    "--max-length",
    type=float,
    default=None,
    help="Maximum video length in minutes per file. If omitted, convert all frames.",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=None,
    help="Max parallel jobs. If omitted, use all available CPU cores.",
)
def main(
    directory: Path,
    vmin: float | None,
    vmax: float | None,
    max_length: float | None,
    jobs: int | None,
) -> None:
    """Convert all CSQ files in a directory to AVI in parallel."""
    root = directory.resolve()
    max_workers = jobs if jobs is not None else default_max_workers()

    csq_files = sorted(root.glob("*.csq"))
    if not csq_files:
        click.echo(f"No .csq files in {root}")
        return

    pending = []
    for f in csq_files:
        out = f.with_suffix(".avi")
        if out.exists():
            click.echo(f"Skipping (exists): {out}", err=True)
            continue
        pending.append((f, out))

    if not pending:
        click.echo("All files already converted.")
        return

    available_workers = default_max_workers()
    used_workers = min(max_workers, len(pending))
    click.echo(f"Available workers: {available_workers}")
    click.echo(f"Converting {len(pending)} file(s) with {used_workers} workers...")
    with ProcessPoolExecutor(max_workers=used_workers) as executor:
        futures = [
            executor.submit(run_conversion_job, inp, out, vmin, vmax, max_length)
            for inp, out in pending
        ]
        with tqdm(total=len(futures), desc="Files converted", unit="file") as pbar:
            for future in as_completed(futures):
                output_path, status, detail = future.result()
                pbar.update(1)
                if status == "done":
                    tqdm.write(f"Done: {output_path}")
                elif status == "skipped":
                    tqdm.write(f"Skipping {output_path}: {detail}")
                else:
                    tqdm.write(f"Error converting {output_path}: {detail}")


if __name__ == "__main__":
    main()
