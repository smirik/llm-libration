#!/usr/bin/env python3
"""
Command Line Interface for LLM Libration package.
"""

import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import click
import numpy as np

from llm_libration import LibrationAnalyzer
from llm_libration.data.plot import create_plots_from_input
from llm_libration.types import ResonanceType
from llm_libration.benchmark import (
    find_png_files,
    map_folder_to_expected_result,
    calculate_metrics,
    save_benchmark_results,
)


@click.group()
@click.version_option()
def main():
    """LLM Libration: AI-powered analysis of resonant angle libration patterns."""
    pass


@main.command()
@click.argument('benchmark_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--provider',
    default='openai',
    type=click.Choice(['openai', 'anthropic', 'openrouter', 'ollama'], case_sensitive=False),
    help='LLM provider to use (default: openai)',
)
@click.option('--model', 'model_name', help='Model name override (optional)')
def benchmark(benchmark_dir: Path, provider: str, model_name: Optional[str]):
    """Run benchmark evaluation on categorized resonance images.

    BENCHMARK_DIR: Path to directory containing categorized subdirectories
    (libration, circulation/non-resonant, transient, controversial)
    with PNG images to analyze.
    """
    start_time = datetime.now()

    click.echo(f"🔬 Starting benchmark evaluation...")
    click.echo(f"📁 Benchmark directory: {benchmark_dir}")
    click.echo(f"🤖 Provider: {provider}")
    if model_name:
        click.echo(f"🔧 Model: {model_name}")
    click.echo(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo()

    # Find all PNG files
    png_files = find_png_files(benchmark_dir)

    if not png_files:
        click.echo("❌ No PNG files found in the benchmark directory!")
        sys.exit(1)

    click.echo(f"🖼️  Found {len(png_files)} PNG files to analyze")
    click.echo()

    # Initialize analyzer
    try:
        if model_name:
            analyzer = LibrationAnalyzer(provider=provider, model_name=model_name)
        else:
            analyzer = LibrationAnalyzer(provider=provider)
    except Exception as e:
        click.echo(f"❌ Failed to initialize analyzer: {e}")
        sys.exit(1)

    # Process each file
    results = []
    success_count = 0

    for i, png_file in enumerate(png_files, 1):
        relative_path = png_file.relative_to(benchmark_dir)

        # Determine expected result based on folder structure
        folder_parts = relative_path.parts
        if len(folder_parts) > 0:
            expected_type = map_folder_to_expected_result(folder_parts[0])
        else:
            expected_type = ResonanceType.CONTROVERSIAL

        click.echo(f"[{i:2d}/{len(png_files)}] Processing: {relative_path}")

        try:
            # Analyze the image
            actual_type = analyzer.analyze_image(png_file)
            success_count += 1

            result = {
                'filename': png_file.name,
                'full_path': str(relative_path),
                'expected_result': expected_type.value,
                'actual_result': actual_type.value,
            }
            results.append(result)

            # Show result with emoji
            match_emoji = "✅" if expected_type == actual_type else "❌"
            click.echo(f"    {match_emoji} Expected: {expected_type.value}, Got: {actual_type.value}")

        except Exception as e:
            click.echo(f"    ❌ Error: {e}")
            # Still add to results with error marker
            result = {
                'filename': png_file.name,
                'full_path': str(relative_path),
                'expected_result': expected_type.value,
                'actual_result': 'error',
            }
            results.append(result)

    click.echo()

    # Calculate metrics
    successful_results = [r for r in results if r['actual_result'] != 'error']
    metrics = calculate_metrics(successful_results)

    # Save results
    try:
        csv_path, json_path = save_benchmark_results(benchmark_dir, results, metrics, provider, model_name, start_time)
        click.echo(f"💾 Results saved to:")
        click.echo(f"   📊 CSV: {csv_path.name}")
        click.echo(f"   📋 Details: {json_path.name}")
        click.echo()
    except Exception as e:
        click.echo(f"⚠️  Warning: Failed to save results: {e}")
        click.echo()

    # Display summary
    end_time = datetime.now()
    duration = end_time - start_time

    click.echo("🏁 BENCHMARK RESULTS")
    click.echo("=" * 50)
    click.echo(f"⏰ Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"🤖 Provider: {provider}")
    click.echo(f"🔧 Model: {model_name or 'default'}")
    click.echo(f"📊 Total Files: {len(png_files)}")
    click.echo(f"✅ Successful: {success_count}")
    click.echo(f"❌ Errors: {len(png_files) - success_count}")
    click.echo(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
    click.echo()

    if successful_results:
        click.echo("📈 CLASSIFICATION METRICS")
        click.echo("-" * 30)
        click.echo(f"True Positives:  {metrics['true_positives']:3d}")
        click.echo(f"True Negatives:  {metrics['true_negatives']:3d}")
        click.echo(f"False Positives: {metrics['false_positives']:3d}")
        click.echo(f"False Negatives: {metrics['false_negatives']:3d}")
        click.echo()
        click.echo(f"Accuracy:  {metrics['accuracy']:.3f}")
        click.echo(f"Precision: {metrics['precision']:.3f}")
        click.echo(f"Recall:    {metrics['recall']:.3f}")
        click.echo(f"F1 Score:  {metrics['f1_score']:.3f}")
    else:
        click.echo("❌ No successful analyses to calculate metrics")


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
