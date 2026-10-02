"""Run command for CLI."""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import click

from llm_libration import LibrationAnalyzer, config
from llm_libration.exceptions import ConfigurationError
from llm_libration.types import ResonanceType, simplify_resonance_type


SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}


def collect_image_files(input_paths: List[Path]) -> List[Path]:
    """Return a unique, sorted list of image files from paths and folders."""
    image_files: List[Path] = []

    for input_path in input_paths:
        if input_path.is_dir():
            for candidate in sorted(p for p in input_path.rglob("*") if p.is_file()):
                if candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.append(candidate)
        else:
            image_files.append(input_path)

    unique_files: List[Path] = []
    seen = set()
    for file_path in image_files:
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)

    return unique_files


def analyze_multiple_images(
    image_paths: List[Path],
    provider: str,
    model_name: Optional[str] = None,
    prompt_template: Optional[str] = None,
    simplified: bool = False,
    processed_pairs: Optional[Set[Tuple[str, str]]] = None,
    results_callback: Optional[Callable[[Dict[str, str]], None]] = None,
    collect_results: bool = True,
) -> List[Dict[str, str]]:
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

    prompt_template = prompt_template or config.prompt_template

    analyzers: Dict[str, LibrationAnalyzer] = {}

    for prov in providers:
        try:
            init_kwargs = {'provider': prov}
            if model_name:
                init_kwargs['model_name'] = model_name
            analyzers[prov] = LibrationAnalyzer(**init_kwargs)
        except Exception as exc:
            click.echo(f"{prov.upper()}: Initialization error - {exc}", err=True)

    if not analyzers:
        click.echo("No providers available for analysis.", err=True)
        return

    click.echo("Selected LLM models for this run:")
    for prov, analyzer in analyzers.items():
        click.echo(f"  - {prov.upper()}: {analyzer.llm_client.model_name}")

    results: List[Dict[str, str]] = []

    def record_result(row: Dict[str, str]) -> None:
        if collect_results:
            results.append(row)
        if results_callback:
            results_callback(row)

    for image_path in image_paths:
        providers_to_process = providers
        resolved_path = image_path.resolve()

        if processed_pairs:
            providers_to_process = [p for p in providers if (p, str(resolved_path)) not in processed_pairs]

        if not providers_to_process:
            continue

        if not image_path.exists():
            click.echo(f"Error: Image file {image_path} not found", err=True)
            continue

        click.echo(f"\nAnalyzing image: {image_path}")
        click.echo("-" * 40)

        for prov in providers_to_process:
            analyzer = analyzers.get(prov)
            if analyzer is None:
                continue
            try:
                result = analyzer.analyze_image(image_path, prompt=prompt_template)
                status: ResonanceType = result.status
                subtype = result.subtype or ""
                if simplified:
                    status = simplify_resonance_type(status)
                click.echo(f"{prov.upper()}: {status.value}")
                record_result(
                    {
                        'provider': prov,
                        'image': str(resolved_path),
                        'status': status.value,
                        'subtype': subtype,
                    }
                )
            except Exception as exc:
                click.echo(f"{prov.upper()}: Error - {exc}")
                record_result(
                    {
                        'provider': prov,
                        'image': str(resolved_path),
                        'status': 'error',
                        'subtype': str(exc),
                    }
                )

    return results


def write_results_to_csv(results: List[Dict[str, str]], output_path: Path, append: bool = False) -> Path:
    """Persist analysis results to CSV, optionally appending to an existing file."""
    output_path = output_path.with_suffix('.csv') if output_path.suffix.lower() != '.csv' else output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()
    mode = 'a' if append and file_exists else 'w'
    include_header = not file_exists or not append

    with output_path.open(mode, newline='', encoding='utf-8') as csvfile:
        fieldnames = ['provider', 'image', 'status', 'subtype']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()
        for row in results:
            writer.writerow(row)

    return output_path


def load_processed_pairs(csv_path: Path) -> Set[Tuple[str, str]]:
    """Load processed (provider, image) pairs from an existing CSV file."""
    processed: Set[Tuple[str, str]] = set()
    if not csv_path.exists():
        return processed

    try:
        with csv_path.open('r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                provider = row.get('provider')
                image = row.get('image')
                if provider and image:
                    processed.add((provider, image))
    except Exception:
        # If the CSV is malformed, treat as empty to avoid blocking progress
        return set()

    return processed


@click.command()
@click.argument('image_files', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    '--provider',
    default='openai',
    type=click.Choice(['openai', 'anthropic', 'openrouter', 'ollama', 'all'], case_sensitive=False),
    help='LLM provider to use (default: openai). Use "all" to try all providers.',
)
@click.option('--model', 'model_name', help='Model name override (optional)')
@click.option(
    '--prompt-env-var',
    default='PROMPT_TEMPLATE',
    show_default=True,
    help='Environment variable that contains the prompt template applied to images.',
)
@click.option(
    '--simplified',
    is_flag=True,
    help='Binary classification helper: use the simplified prompt by default and map transient to resonant.',
)
@click.option(
    '--batch-size',
    default=100,
    show_default=True,
    type=click.IntRange(1),
    help='Number of images to process per batch.',
)
@click.option(
    '--resume',
    is_flag=True,
    help='Resume from the last saved batch in the CSV output file.',
)
@click.option(
    '--output-csv',
    type=click.Path(path_type=Path),
    help='Path to save analysis results as CSV. Automatically generated when a directory is provided.',
)
def run(
    image_files: tuple,
    provider: str,
    model_name: Optional[str],
    prompt_env_var: str,
    simplified: bool,
    batch_size: int,
    resume: bool,
    output_csv: Optional[Path],
):
    """Run libration analysis on one or more image files.

    IMAGE_FILES: One or more paths to image files or directories to analyze.

    Output format: PROVIDER: status
    Example: OPENAI: resonant
    """
    raw_paths = [Path(f) for f in image_files]
    image_paths = collect_image_files(raw_paths)

    if not image_paths:
        click.echo("No image files found to analyze.", err=True)
        sys.exit(1)

    effective_prompt_var = (prompt_env_var or "PROMPT_TEMPLATE").strip() or "PROMPT_TEMPLATE"
    if simplified and effective_prompt_var.upper() == "PROMPT_TEMPLATE":
        effective_prompt_var = "PROMPT_TEMPLATE_SIMPLIFIED"

    try:
        prompt_template = config.get_prompt_template(effective_prompt_var)
    except ConfigurationError as exc:
        click.echo(f"❌ {exc}", err=True)
        sys.exit(1)

    target_csv = output_csv
    provided_directory = any(path.is_dir() for path in raw_paths)
    if target_csv is None and provided_directory:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_directory = next((path for path in raw_paths if path.is_dir()), Path.cwd())
        target_csv = target_directory / f"libration_run_results_{timestamp}.csv"

    if resume and target_csv is None:
        click.echo("--resume requires an output CSV path (either via --output-csv or directory input).", err=True)
        sys.exit(1)

    processed_pairs: Set[Tuple[str, str]] = set()
    if resume and target_csv:
        if not target_csv.exists():
            click.echo(f"Resume requested but CSV file not found: {target_csv}", err=True)
            sys.exit(1)
        processed_pairs = load_processed_pairs(target_csv)

    csv_initialized = False
    first_append_mode = bool(resume and target_csv and target_csv.exists())

    total_images = len(image_paths)
    for start in range(0, total_images, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_results: List[Dict[str, str]] = []

        try:
            analyze_multiple_images(
                batch_paths,
                provider.lower(),
                model_name,
                prompt_template,
                simplified=simplified,
                processed_pairs=processed_pairs if processed_pairs else None,
                results_callback=batch_results.append if target_csv else None,
                collect_results=False,
            )
        except Exception as exc:
            click.echo(f"Analysis failed: {exc}", err=True)
            sys.exit(1)

        if target_csv and batch_results:
            append_mode = first_append_mode if not csv_initialized else True
            try:
                saved_path = write_results_to_csv(batch_results, target_csv, append=append_mode)
                csv_initialized = True
                processed_pairs.update((row['provider'], row['image']) for row in batch_results)
                first_append_mode = False
                click.echo(f"\nBatch saved to CSV: {saved_path}")
            except Exception as exc:
                click.echo(f"Failed to write CSV results: {exc}", err=True)
                sys.exit(1)
