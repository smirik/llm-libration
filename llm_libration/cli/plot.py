"""Plot command for CLI."""

import sys
from pathlib import Path
from typing import Optional
import click
import numpy as np

from llm_libration.data.plot import create_plots_from_input


@click.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--x-column', default='times', help='Column name for x-axis data (default: times)')
@click.option('--y-column', default='angle', help='Column name for y-axis data (default: angle)')
@click.option('--output-file', type=click.Path(path_type=Path), help='Output PNG file path (only for single file input)')
@click.option('--x-min', default=0.0, type=float, help='Minimum x-axis value (default: 0)')
@click.option('--x-max', default=100000.0, type=float, help='Maximum x-axis value (default: 100000)')
@click.option('--y-min', default=0.0, type=float, help='Minimum y-axis value (default: 0)')
@click.option('--y-max', default=2 * np.pi, type=float, help='Maximum y-axis value (default: 2*pi)')
def plot(
    input_path: Path, x_column: str, y_column: str, output_file: Optional[Path], x_min: float, x_max: float, y_min: float, y_max: float
):
    """Create plot(s) from CSV data.

    INPUT_PATH: Path to a CSV file or folder containing CSV files.
    If a folder is provided, plots will be created for all CSV files found recursively.
    """
    try:
        result = create_plots_from_input(
            input_path=input_path,
            x_column=x_column,
            y_column=y_column,
            output_file=output_file,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        if isinstance(result, str):
            click.echo(f"Plot created successfully: {result}")
        else:
            click.echo(f"Batch processing completed. {len(result)} plots created.")
    except Exception as e:
        click.echo(f"Error creating plot(s): {e}", err=True)
        sys.exit(1)
