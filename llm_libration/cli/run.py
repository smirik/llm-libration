"""Run command for CLI."""

import sys
from pathlib import Path
from typing import List, Optional
import click

from llm_libration import LibrationAnalyzer


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
                click.echo(f"{prov.upper()}: {result.status} ({result.subtype})")
            except Exception as e:
                click.echo(f"{prov.upper()}: Error - {e}")


@click.command()
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

    Output format: PROVIDER: status (subtype)
    Example: OPENAI: resonant (apocentric libration)
    """
    image_paths = [Path(f) for f in image_files]

    try:
        analyze_multiple_images(image_paths, provider.lower(), model_name)
    except Exception as e:
        click.echo(f"Analysis failed: {e}", err=True)
        sys.exit(1)
