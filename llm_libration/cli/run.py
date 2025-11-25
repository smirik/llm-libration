"""Run command for CLI."""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import click

from llm_libration import LibrationAnalyzer, config
from llm_libration.exceptions import ConfigurationError
from llm_libration.types import ResonanceType, simplify_resonance_type


def analyze_multiple_images(
    image_paths: List[Path],
    provider: str,
    model_name: Optional[str] = None,
    prompt_template: Optional[str] = None,
    simplified: bool = False,
) -> None:
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

    for image_path in image_paths:
        if not image_path.exists():
            click.echo(f"Error: Image file {image_path} not found", err=True)
            continue

        click.echo(f"\nAnalyzing image: {image_path}")
        click.echo("-" * 40)

        for prov in providers:
            analyzer = analyzers.get(prov)
            if analyzer is None:
                continue
            try:
                result = analyzer.analyze_image(image_path, prompt=prompt_template)
                status: ResonanceType = result.status
                if simplified:
                    status = simplify_resonance_type(status)
                click.echo(f"{prov.upper()}: {status.value}")
            except Exception as exc:
                click.echo(f"{prov.upper()}: Error - {exc}")


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
def run(image_files: tuple, provider: str, model_name: Optional[str], prompt_env_var: str, simplified: bool):
    """Run libration analysis on one or more image files.

    IMAGE_FILES: One or more paths to image files to analyze.

    Output format: PROVIDER: status
    Example: OPENAI: resonant
    """
    image_paths = [Path(f) for f in image_files]

    effective_prompt_var = (prompt_env_var or "PROMPT_TEMPLATE").strip() or "PROMPT_TEMPLATE"
    if simplified and effective_prompt_var.upper() == "PROMPT_TEMPLATE":
        effective_prompt_var = "PROMPT_TEMPLATE_SIMPLIFIED"

    try:
        prompt_template = config.get_prompt_template(effective_prompt_var)
    except ConfigurationError as exc:
        click.echo(f"❌ {exc}", err=True)
        sys.exit(1)

    try:
        analyze_multiple_images(image_paths, provider.lower(), model_name, prompt_template, simplified=simplified)
    except Exception as exc:
        click.echo(f"Analysis failed: {exc}", err=True)
        sys.exit(1)
