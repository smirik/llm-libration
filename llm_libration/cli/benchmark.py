"""Benchmark command for CLI."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from llm_libration import LibrationAnalyzer
from llm_libration.benchmark import (
    calculate_metrics,
    calculate_relaxed_metrics,
    find_png_files,
    map_folder_to_expected_result,
    save_benchmark_results,
)
from llm_libration.config import config
from llm_libration.exceptions import ConfigurationError
from llm_libration.types import ResonanceType, simplify_resonance_type


@click.command()
@click.argument('benchmark_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--provider',
    default='openai',
    type=click.Choice(['openai', 'anthropic', 'openrouter', 'ollama'], case_sensitive=False),
    help='LLM provider to use (default: openai)',
)
@click.option('--model', 'model_name', help='Model name override (optional)')
@click.option(
    '--prompt-env-var',
    default='PROMPT_TEMPLATE',
    show_default=True,
    help='Environment variable that stores the prompt template.',
)
@click.option(
    '--simplified',
    is_flag=True,
    help='Simplify benchmark labels so transient+libration count as resonant and everything else as non-resonant.',
)
def benchmark(benchmark_dir: Path, provider: str, model_name: Optional[str], prompt_env_var: str, simplified: bool):
    """Run benchmark evaluation on categorized resonance images.

    BENCHMARK_DIR: Path to directory containing categorized subdirectories
    (libration, circulation/non-resonant, transient, controversial)
    with PNG images to analyze.
    """
    # Configure logging to show retry warnings
    logging.basicConfig(
        level=logging.WARNING,
        format='    ⚠️  %(message)s'
    )

    start_time = datetime.now()

    prompt_variable = (prompt_env_var or "PROMPT_TEMPLATE").strip() or "PROMPT_TEMPLATE"
    if simplified and prompt_variable.upper() == "PROMPT_TEMPLATE":
        prompt_variable = "PROMPT_TEMPLATE_SIMPLIFIED"

    try:
        prompt_template = config.get_prompt_template(prompt_variable)
        init_kwargs = {'provider': provider}
        if model_name:
            init_kwargs['model_name'] = model_name
        analyzer = LibrationAnalyzer(**init_kwargs)
    except ConfigurationError as exc:
        click.echo(f"❌ {exc}")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"❌ Failed to initialize analyzer: {exc}")
        sys.exit(1)

    resolved_provider = analyzer.llm_client.provider
    resolved_model = analyzer.llm_client.model_name

    click.echo("🔬 Starting benchmark evaluation...")
    click.echo(f"📁 Benchmark directory: {benchmark_dir}")
    click.echo(f"🤖 Provider: {resolved_provider}")
    click.echo(f"🔧 Model: {resolved_model}")
    click.echo(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"📝 Prompt variable: {prompt_variable}")
    click.echo(f"🧮 Mode: {'simplified (binary resonant/non-resonant)' if simplified else 'full spectrum'}")
    click.echo()

    png_files = find_png_files(benchmark_dir)

    if not png_files:
        click.echo("❌ No PNG files found in the benchmark directory!")
        sys.exit(1)

    click.echo(f"🖼️  Found {len(png_files)} PNG files to analyze")
    click.echo()

    results = []
    success_count = 0

    for i, png_file in enumerate(png_files, 1):
        relative_path = png_file.relative_to(benchmark_dir)

        folder_parts = relative_path.parts
        if folder_parts:
            expected_type = map_folder_to_expected_result(folder_parts[0], simplified=simplified)
        else:
            expected_type = ResonanceType.NON_RESONANT if simplified else ResonanceType.CONTROVERSIAL

        click.echo(f"[{i:2d}/{len(png_files)}] Processing: {relative_path}")

        try:
            full_result = analyzer.analyze_image(png_file, prompt=prompt_template)
            actual_type = full_result.status
            if simplified:
                actual_type = simplify_resonance_type(actual_type)
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

        except Exception as exc:
            click.echo(f"    ❌ Error: {exc}")
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
    relaxed_metrics = calculate_relaxed_metrics(successful_results)

    # Save results
    try:
        csv_path, json_path = save_benchmark_results(benchmark_dir, results, metrics, resolved_provider, resolved_model, start_time)
        click.echo("💾 Results saved to:")
        click.echo(f"   📊 CSV: {csv_path.name}")
        click.echo(f"   📋 Details: {json_path.name}")
        click.echo()
    except Exception as exc:
        click.echo(f"⚠️  Warning: Failed to save results: {exc}")
        click.echo()

    end_time = datetime.now()
    duration = end_time - start_time

    click.echo("🏁 BENCHMARK RESULTS")
    click.echo("=" * 50)
    click.echo(f"⏰ Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"🤖 Provider: {resolved_provider}")
    click.echo(f"🔧 Model: {resolved_model}")
    click.echo(f"📊 Total Files: {len(png_files)}")
    click.echo(f"✅ Successful: {success_count}")
    click.echo(f"❌ Errors: {len(png_files) - success_count}")
    click.echo(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
    click.echo()

    if successful_results:
        click.echo("📈 STRICT CLASSIFICATION METRICS")
        click.echo("-" * 40)
        click.echo(f"True Positives:  {metrics['true_positives']:3d}")
        click.echo(f"True Negatives:  {metrics['true_negatives']:3d}")
        click.echo(f"False Positives: {metrics['false_positives']:3d}")
        click.echo(f"False Negatives: {metrics['false_negatives']:3d}")
        click.echo()
        click.echo(f"Accuracy:  {metrics['accuracy']:.3f}")
        click.echo(f"Precision: {metrics['precision']:.3f}")
        click.echo(f"Recall:    {metrics['recall']:.3f}")
        click.echo(f"F1 Score:  {metrics['f1_score']:.3f}")
        click.echo()

        click.echo("📈 RELAXED CLASSIFICATION METRICS")
        click.echo("-" * 40)
        click.echo("Acceptance criteria:")
        click.echo("  • resonant → accepts: resonant, controversial")
        click.echo("  • non-resonant → accepts: non-resonant, controversial, transient")
        click.echo("  • transient → accepts: transient, resonant, controversial")
        click.echo("  • controversial → accepts: any")
        click.echo()
        click.echo(f"Correct Predictions:   {relaxed_metrics['correct_predictions']:3d}")
        click.echo(f"Incorrect Predictions: {relaxed_metrics['incorrect_predictions']:3d}")
        click.echo(f"Relaxed Accuracy:      {relaxed_metrics['accuracy']:.3f}")
    else:
        click.echo("❌ No successful analyses to calculate metrics")
