"""Tests for the run CLI command."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from llm_libration.cli import main
from llm_libration.cli.run import analyze_multiple_images
from llm_libration.types import ResonanceType


class TestRunCLI:
    """Test cases for run CLI command."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture(autouse=True)
    def mock_prompt_template(self):
        """Ensure CLI always sees a prompt template."""
        with patch('llm_libration.cli.run.config.get_prompt_template', return_value='dummy prompt') as mock_get:
            yield mock_get

    @pytest.fixture
    def sample_image_file(self):
        """Create a temporary PNG file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Write a minimal valid PNG header
            f.write(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            image_path = f.name

        yield image_path

        # Cleanup
        os.unlink(image_path)

    def test_run_help(self, runner):
        """Test the run command help."""
        result = runner.invoke(main, ['run', '--help'])
        assert result.exit_code == 0
        assert 'Run libration analysis' in result.output
        assert '--provider' in result.output
        assert '--model' in result.output

    def test_run_single_image(self, runner, sample_image_file):
        """Test run command with single image."""
        with patch('llm_libration.cli.run.analyze_multiple_images') as mock_analyze:
            result = runner.invoke(main, ['run', sample_image_file])

            assert result.exit_code == 0
            mock_analyze.assert_called_once_with(
                [Path(sample_image_file)], 'openai', None, 'dummy prompt', simplified=False
            )

    def test_run_multiple_images(self, runner, sample_image_file):
        """Test run command with multiple images."""
        # Create a second image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            image_path2 = f.name

        try:
            with patch('llm_libration.cli.run.analyze_multiple_images') as mock_analyze:
                result = runner.invoke(main, ['run', sample_image_file, image_path2, '--provider', 'anthropic'])

                assert result.exit_code == 0
                mock_analyze.assert_called_once_with(
                    [Path(sample_image_file), Path(image_path2)],
                    'anthropic',
                    None,
                    'dummy prompt',
                    simplified=False,
                )
        finally:
            os.unlink(image_path2)

    def test_run_with_custom_model(self, runner, sample_image_file):
        """Test run command with custom model."""
        with patch('llm_libration.cli.run.analyze_multiple_images') as mock_analyze:
            result = runner.invoke(main, ['run', sample_image_file, '--provider', 'openai', '--model', 'gpt-4-vision'])

            assert result.exit_code == 0
            mock_analyze.assert_called_once_with(
                [Path(sample_image_file)], 'openai', 'gpt-4-vision', 'dummy prompt', simplified=False
            )

    def test_run_prompt_env_option(self, runner, sample_image_file, mock_prompt_template):
        """Ensure custom prompt variable name is honored."""
        with patch('llm_libration.cli.run.analyze_multiple_images') as mock_analyze:
            result = runner.invoke(
                main, ['run', sample_image_file, '--prompt-env-var', 'ALT_PROMPT']
            )

            assert result.exit_code == 0
            mock_prompt_template.assert_called_with('ALT_PROMPT')
            mock_analyze.assert_called_once_with(
                [Path(sample_image_file)], 'openai', None, 'dummy prompt', simplified=False
            )

    def test_run_simplified_defaults_prompt(self, runner, sample_image_file, mock_prompt_template):
        """Simplified flag selects the binary prompt when no custom env var is provided."""
        mock_prompt_template.return_value = 'binary prompt'
        with patch('llm_libration.cli.run.analyze_multiple_images') as mock_analyze:
            result = runner.invoke(main, ['run', sample_image_file, '--simplified'])

            assert result.exit_code == 0
            mock_prompt_template.assert_called_with('PROMPT_TEMPLATE_SIMPLIFIED')
            mock_analyze.assert_called_once_with(
                [Path(sample_image_file)], 'openai', None, 'binary prompt', simplified=True
            )

    def test_run_error_handling(self, runner, sample_image_file):
        """Test run command error handling."""
        with patch('llm_libration.cli.run.analyze_multiple_images') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")

            result = runner.invoke(main, ['run', sample_image_file])

            assert result.exit_code == 1
            assert 'Analysis failed: Analysis failed' in result.output

    def test_run_image_not_found(self, runner):
        """Test run command with non-existent image."""
        result = runner.invoke(main, ['run', 'nonexistent.png'])
        assert result.exit_code == 2  # Click's file not found exit code

    def test_analyze_multiple_images_single_provider(self, sample_image_file):
        """Test analyze_multiple_images function with single provider."""
        with patch('llm_libration.cli.run.LibrationAnalyzer') as mock_analyzer_class, patch(
            'llm_libration.cli.run.click.echo'
        ) as mock_echo:
            from llm_libration.llm.schema import LibrationAnalysisResult, ResonantSubtype

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = LibrationAnalysisResult(
                status=ResonanceType.RESONANT, subtype=ResonantSubtype.APOCENTRIC_LIBRATION
            )
            mock_analyzer_class.return_value = mock_analyzer

            analyze_multiple_images([Path(sample_image_file)], 'openai', None, 'prompt-here', False)

            mock_analyzer_class.assert_called_once_with(provider='openai')
            mock_analyzer.analyze_image.assert_called_once()

    def test_analyze_multiple_images_all_providers(self, sample_image_file):
        """Test analyze_multiple_images function with all providers."""
        with patch('llm_libration.cli.run.LibrationAnalyzer') as mock_analyzer_class, patch(
            'llm_libration.cli.run.click.echo'
        ) as mock_echo:
            from llm_libration.llm.schema import LibrationAnalysisResult, ResonantSubtype

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = LibrationAnalysisResult(
                status=ResonanceType.RESONANT, subtype=ResonantSubtype.APOCENTRIC_LIBRATION
            )
            mock_analyzer_class.return_value = mock_analyzer

            analyze_multiple_images([Path(sample_image_file)], 'all', None, 'prompt-here', False)

            # Should be called 4 times (once for each provider)
            assert mock_analyzer_class.call_count == 4
            assert mock_analyzer.analyze_image.call_count == 4

    def test_analyze_multiple_images_with_custom_model(self, sample_image_file):
        """Test analyze_multiple_images function with custom model."""
        with patch('llm_libration.cli.run.LibrationAnalyzer') as mock_analyzer_class, patch(
            'llm_libration.cli.run.click.echo'
        ) as mock_echo:
            from llm_libration.llm.schema import LibrationAnalysisResult, ResonantSubtype

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = LibrationAnalysisResult(
                status=ResonanceType.RESONANT, subtype=ResonantSubtype.APOCENTRIC_LIBRATION
            )
            mock_analyzer_class.return_value = mock_analyzer

            analyze_multiple_images([Path(sample_image_file)], 'anthropic', 'claude-sonnet-4', 'prompt-here', False)

            mock_analyzer_class.assert_called_once_with(provider='anthropic', model_name='claude-sonnet-4')

    def test_analyze_multiple_images_simplified_mapping(self, sample_image_file):
        """Simplified mode collapses transient to resonant."""
        with patch('llm_libration.cli.run.LibrationAnalyzer') as mock_analyzer_class, patch(
            'llm_libration.cli.run.click.echo'
        ) as mock_echo:
            from llm_libration.llm.schema import LibrationAnalysisResult, ResonantSubtype

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.return_value = LibrationAnalysisResult(
                status=ResonanceType.TRANSIENT, subtype=ResonantSubtype.CLEAR_LIBRATION
            )
            mock_analyzer_class.return_value = mock_analyzer

            analyze_multiple_images([Path(sample_image_file)], 'openai', None, 'prompt', True)

            # Should print resonant even though LLM returned transient
            simplified_outputs = [call for call in mock_echo.call_args_list if 'OPENAI' in str(call)]
            assert any('resonant' in str(call).lower() for call in simplified_outputs)

    def test_analyze_multiple_images_error_handling(self, sample_image_file):
        """Test analyze_multiple_images function error handling."""
        with patch('llm_libration.cli.run.LibrationAnalyzer') as mock_analyzer_class, patch(
            'llm_libration.cli.run.click.echo'
        ) as mock_echo:

            mock_analyzer = MagicMock()
            mock_analyzer.analyze_image.side_effect = Exception("Analysis error")
            mock_analyzer_class.return_value = mock_analyzer

            # Should not raise exception, just print error
            analyze_multiple_images([Path(sample_image_file)], 'openai', None, 'prompt-here', False)

            # Check that error was echoed
            error_calls = [call for call in mock_echo.call_args_list if 'Error' in str(call)]
            assert len(error_calls) > 0

    def test_analyze_multiple_images_nonexistent_file(self):
        """Test analyze_multiple_images function with non-existent file."""
        with patch('llm_libration.cli.run.click.echo') as mock_echo:

            analyze_multiple_images([Path('/nonexistent/file.png')], 'openai', None, 'prompt-here', False)

            # Check that error was echoed for missing file
            error_calls = [call for call in mock_echo.call_args_list if 'not found' in str(call)]
            assert len(error_calls) > 0
