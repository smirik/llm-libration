"""
Command Line Interface for LLM Libration package.

This module provides a refactored CLI structure with individual command modules.
"""

import click

from .benchmark import benchmark
from .plot import plot
from .run import run


@click.group()
@click.version_option()
def main():
    """LLM Libration: AI-powered analysis of resonant angle libration patterns."""
    pass


# Register commands
main.add_command(benchmark)
main.add_command(plot)
main.add_command(run)


if __name__ == '__main__':
    main()
