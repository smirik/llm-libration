#!/usr/bin/env python3
"""
Plotting utilities for creating resonant angle plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional


def create_plot(
    input_file: Union[str, Path],
    x_column: str = 'times',
    y_column: str = 'angle',
    output_file: Optional[Union[str, Path]] = None,
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
    plt.figure(figsize=(10, 5))
    plt.plot(data[x_column], data[y_column], linestyle='', marker=',', color='black')

    # Set y-axis limits
    plt.ylim(y_min, y_max)

    # Remove ticks and legend
    plt.xticks([])
    plt.yticks([])
    plt.gca().legend().set_visible(False) if plt.gca().get_legend() else None

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return str(output_path.absolute())
