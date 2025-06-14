"""Benchmark command for CLI."""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import click

from llm_libration import LibrationAnalyzer
from llm_libration.types import ResonanceType
from llm_libration.benchmark import (
    find_png_files,
    map_folder_to_expected_result,
    calculate_metrics,
    save_benchmark_results,
)


@click.command()
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

    png_files = find_png_files(benchmark_dir)

    if not png_files:
        click.echo("❌ No PNG files found in the benchmark directory!")
        sys.exit(1)

    click.echo(f"🖼️  Found {len(png_files)} PNG files to analyze")
    click.echo()

    try:
        if model_name:
            analyzer = LibrationAnalyzer(provider=provider, model_name=model_name)
        else:
            analyzer = LibrationAnalyzer(provider=provider)
    except Exception as e:
        click.echo(f"❌ Failed to initialize analyzer: {e}")
        sys.exit(1)

    results = []
    success_count = 0

    for i, png_file in enumerate(png_files, 1):
        relative_path = png_file.relative_to(benchmark_dir)

        folder_parts = relative_path.parts
        if len(folder_parts) > 0:
            expected_type = map_folder_to_expected_result(folder_parts[0])
        else:
            expected_type = ResonanceType.CONTROVERSIAL

        click.echo(f"[{i:2d}/{len(png_files)}] Processing: {relative_path}")

        try:
            actual_type = analyzer.get_resonance_type(png_file)
            success_count += 1

            result = {
                'filename': png_file.name,
                'full_path': str(relative_path),
                'expected_result': expected_type.value,
                'actual_result': actual_type.value,
            }
            results.append(result)

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
