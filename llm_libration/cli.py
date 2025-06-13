#!/usr/bin/env python3
"""
Command Line Interface for LLM Libration package.
"""

import sys
from pathlib import Path
from typing import List, Optional
import click
import numpy as np

from llm_libration import LibrationAnalyzer
from llm_libration.data.plot import create_plots_from_input


@click.group()
@click.version_option()
def main():
    """LLM Libration: AI-powered analysis of resonant angle libration patterns."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--x-column', default='times', help='Column name for x-axis data (default: times)')
@click.option('--y-column', default='angle', help='Column name for y-axis data (default: angle)')
@click.option('--output-file', type=click.Path(path_type=Path), help='Output PNG file path (only for single file input)')
@click.option('--y-min', default=0.0, type=float, help='Minimum y-axis value (default: 0)')
@click.option('--y-max', default=2 * np.pi, type=float, help='Maximum y-axis value (default: 2*pi)')
def plot(input_path: Path, x_column: str, y_column: str, output_file: Optional[Path], y_min: float, y_max: float):
    """Create plot(s) from CSV data.

    INPUT_PATH: Path to a CSV file or folder containing CSV files.
    If a folder is provided, plots will be created for all CSV files found recursively.
    """
    try:
        result = create_plots_from_input(
            input_path=input_path, x_column=x_column, y_column=y_column, output_file=output_file, y_min=y_min, y_max=y_max
        )

        if isinstance(result, str):
            click.echo(f"Plot created successfully: {result}")
        else:
            click.echo(f"Batch processing completed. {len(result)} plots created.")
    except Exception as e:
        click.echo(f"Error creating plot(s): {e}", err=True)
        sys.exit(1)


def analyze_multiple_images(image_paths: List[Path], provider: str, model_name: Optional[str] = None) -> None:
    """
    Analyze multiple images with specified provider(s).

    Args:
        image_paths: List of paths to image files
        provider: Provider name or 'all' for all providers
        model_name: Optional model name override
    """
    if provider == 'all':
        providers = ["openai", "anthropic", "openrouter", "ollama"]
    else:
        providers = [provider]

    for image_path in image_paths:
        if not image_path.exists():
            click.echo(f"Error: Image file {image_path} not found", err=True)
            continue

        click.echo(f"\nAnalyzing image: {image_path}")
        click.echo("-" * 40)

        for prov in providers:
            try:
                if model_name:
                    analyzer = LibrationAnalyzer(provider=prov, model_name=model_name)
                else:
                    analyzer = LibrationAnalyzer(provider=prov)
                result = analyzer.analyze_image(image_path)
                click.echo(f"{prov.upper()}: {result.value}")
            except Exception as e:
                click.echo(f"{prov.upper()}: Error - {e}")


@main.command()
@click.argument('image_files', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    '--provider',
    default='openai',
    type=click.Choice(['openai', 'anthropic', 'openrouter', 'ollama', 'all'], case_sensitive=False),
    help='LLM provider to use (default: openai). Use "all" to try all providers.',
)
@click.option('--model', 'model_name', help='Model name override (optional)')
def run(image_files: tuple, provider: str, model_name: Optional[str]):
    """Run libration analysis on one or more image files.

    IMAGE_FILES: One or more paths to image files to analyze.
    """
    image_paths = [Path(f) for f in image_files]

    try:
        analyze_multiple_images(image_paths, provider.lower(), model_name)
    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
