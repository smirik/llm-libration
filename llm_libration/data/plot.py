#!/usr/bin/env python3
"""
Plotting utilities for creating resonant angle plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, List
import click


def create_plot(
    input_file: Union[str, Path],
    x_column: str = 'times',
    y_column: str = 'angle',
    output_file: Optional[Union[str, Path]] = None,
    x_min: float = 0,
    x_max: float = 100000,
    y_min: float = 0,
    y_max: float = 2 * np.pi,
) -> str:
    """
    Create a simple scatter plot from CSV data.

    Args:
        input_file: Path to the input CSV file
        x_column: Name of the column to use for x-axis (default: 'times')
        y_column: Name of the column to use for y-axis (default: 'angle')
        output_file: Path for the output PNG file (default: same directory and name as input but with .png extension)
        x_min: Minimum value for x-axis (default: 0)
        x_max: Maximum value for x-axis (default: 100000)
        y_min: Minimum value for y-axis (default: 0)
        y_max: Maximum value for y-axis (default: 2*pi)

    Returns:
        str: Full path to the created PNG file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If required columns are missing from the CSV
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Set default output file path
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}.png"
    else:
        output_path = Path(output_file)

    # Read the CSV data
    try:
        data = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Validate columns exist
    if x_column not in data.columns:
        raise ValueError(f"Column '{x_column}' not found in CSV. Available columns: {list(data.columns)}")
    if y_column not in data.columns:
        raise ValueError(f"Column '{y_column}' not found in CSV. Available columns: {list(data.columns)}")

    # Create the plot
    plt.figure(figsize=(10, 3))
    plt.plot(data[x_column], data[y_column], linestyle='', marker='.', color='black', markersize=1)

    # Set axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Remove ticks and legend
    plt.xticks([])
    plt.yticks([])
    plt.gca().legend().set_visible(False) if plt.gca().get_legend() else None

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return str(output_path.absolute())


def find_csv_files(directory: Path) -> List[Path]:
    """
    Recursively find all CSV files in a directory and its subdirectories.

    Args:
        directory: Path to the directory to search

    Returns:
        List of Path objects for all CSV files found

    Raises:
        FileNotFoundError: If the directory doesn't exist
        NotADirectoryError: If the path is not a directory
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    csv_files = []
    for csv_file in directory.rglob("*.csv"):
        if csv_file.is_file():
            csv_files.append(csv_file)

    return sorted(csv_files)


def create_plots_from_folder(
    input_folder: Union[str, Path],
    x_column: str = 'times',
    y_column: str = 'angle',
    x_min: float = 0,
    x_max: float = 100000,
    y_min: float = 0,
    y_max: float = 2 * np.pi,
    verbose: bool = True,
) -> List[str]:
    """
    Create plots for all CSV files in a folder and its subfolders.

    Args:
        input_folder: Path to the folder containing CSV files
        x_column: Name of the column to use for x-axis (default: 'times')
        y_column: Name of the column to use for y-axis (default: 'angle')
        x_min: Minimum value for x-axis (default: 0)
        x_max: Maximum value for x-axis (default: 100000)
        y_min: Minimum value for y-axis (default: 0)
        y_max: Maximum value for y-axis (default: 2*pi)
        verbose: Whether to print progress messages (default: True)

    Returns:
        List of paths to the created PNG files

    Raises:
        FileNotFoundError: If the input folder doesn't exist
        NotADirectoryError: If the input path is not a directory
    """
    folder_path = Path(input_folder)

    # Find all CSV files
    csv_files = find_csv_files(folder_path)

    if not csv_files:
        if verbose:
            click.echo(f"No CSV files found in {folder_path}")
        return []

    if verbose:
        click.echo(f"Found {len(csv_files)} CSV file(s) in {folder_path}")

    created_plots = []
    failed_plots = []

    for csv_file in csv_files:
        try:
            if verbose:
                try:
                    relative_path = csv_file.relative_to(folder_path.absolute())
                    click.echo(f"Processing: {relative_path}")
                except ValueError:
                    click.echo(f"Processing: {csv_file.name}")

            plot_path = create_plot(
                input_file=csv_file,
                x_column=x_column,
                y_column=y_column,
                output_file=None,  # Use default (same directory as CSV)
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            created_plots.append(plot_path)

            if verbose:
                try:
                    relative_path = Path(plot_path).relative_to(folder_path.absolute())
                    click.echo(f"  ✅ Created: {relative_path}")
                except ValueError:
                    # Fallback if relative path calculation fails
                    click.echo(f"  ✅ Created: {Path(plot_path).name}")

        except Exception as e:
            failed_plots.append((csv_file, str(e)))
            if verbose:
                click.echo(f"  ❌ Failed: {e}")

    if verbose:
        click.echo(f"\n📊 Summary:")
        click.echo(f"  • Successfully created: {len(created_plots)} plots")
        if failed_plots:
            click.echo(f"  • Failed: {len(failed_plots)} plots")
            for failed_file, error in failed_plots:
                click.echo(f"    - {failed_file.name}: {error}")

    return created_plots


def create_plots_from_input(
    input_path: Union[str, Path],
    x_column: str = 'times',
    y_column: str = 'angle',
    output_file: Optional[Union[str, Path]] = None,
    x_min: float = 0,
    x_max: float = 100000,
    y_min: float = 0,
    y_max: float = 2 * np.pi,
) -> Union[str, List[str]]:
    """
    Create plot(s) from either a CSV file or a folder containing CSV files.

    Args:
        input_path: Path to a CSV file or folder containing CSV files
        x_column: Name of the column to use for x-axis (default: 'times')
        y_column: Name of the column to use for y-axis (default: 'angle')
        output_file: Path for the output PNG file (only used for single file input)
        x_min: Minimum value for x-axis (default: 0)
        x_max: Maximum value for x-axis (default: 100000)
        y_min: Minimum value for y-axis (default: 0)
        y_max: Maximum value for y-axis (default: 2*pi)

    Returns:
        For single file: str (path to created PNG)
        For folder: List[str] (paths to all created PNGs)

    Raises:
        FileNotFoundError: If the input path doesn't exist
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_file():
        return create_plot(
            input_file=path,
            x_column=x_column,
            y_column=y_column,
            output_file=output_file,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
    elif path.is_dir():
        if output_file is not None:
            click.echo("Warning: --output-file option ignored when processing folders")

        return create_plots_from_folder(
            input_folder=path, x_column=x_column, y_column=y_column, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, verbose=True
        )
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")
