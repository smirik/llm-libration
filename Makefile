.PHONY: help install dev-install test test-verbose coverage format lint clean clean-all check docs example
.DEFAULT_GOAL := help

# Variables
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
BLACK := .venv/bin/black
FLAKE8 := .venv/bin/flake8

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Create virtual environment and install dependencies
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "✅ Installation complete! Activate with: source .venv/bin/activate"

dev-install: ## Install package in development mode (assumes .venv exists)
	$(PIP) install -e ".[dev]"

test: ## Run tests
	$(PYTEST)

test-verbose: ## Run tests with verbose output
	$(PYTEST) -v

coverage: ## Run tests with coverage report
	$(PYTEST) --cov=llm_libration --cov-report=html --cov-report=term

format: ## Format code with black
	$(BLACK) llm_libration tests example.py

format-check: ## Check if code formatting is correct
	$(BLACK) --check llm_libration tests example.py

lint: ## Run linting with flake8
	$(FLAKE8) llm_libration tests

check: format-check lint test ## Run all code quality checks

clean: ## Clean up temporary files
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf llm_libration.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Clean everything including virtual environment
	rm -rf .venv

reinstall: clean-all install ## Clean everything and reinstall

example: ## Run example script (requires image path: make example IMAGE=path/to/image.png)
	@if [ -z "$(IMAGE)" ]; then \
		echo "❌ Please provide an image path: make example IMAGE=path/to/image.png"; \
		exit 1; \
	fi
	$(PYTHON) example.py $(IMAGE)

docs: ## Generate documentation (placeholder for future use)
	@echo "📚 Documentation generation not yet implemented"

env-info: ## Show environment information
	@echo "Python version:"
	@$(PYTHON) --version
	@echo "\nInstalled packages:"
	@$(PIP) list
	@echo "\nPackage info:"
	@$(PIP) show llm-libration

deps-update: ## Update all dependencies to latest versions
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[dev]"

deps-list: ## List all dependencies
	$(PIP) list

build: ## Build the package
	$(PIP) install build
	$(PYTHON) -m build

publish-test: build ## Publish to Test PyPI
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PIP) install twine
	$(PYTHON) -m twine upload dist/*

init-git: ## Initialize git repository and make first commit
	git init
	git add .
	git commit -m "Initial commit: llm-libration package"
	@echo "✅ Git repository initialized"

# Development workflow targets
dev: format lint test ## Run complete development workflow

quick-test: ## Run tests without coverage (faster)
	$(PYTEST) --tb=short

# Environment management
activate: ## Show activation command
	@echo "To activate the virtual environment, run:"
	@echo "source .venv/bin/activate"

status: ## Show project status
	@echo "🔍 Project Status:"
	@echo "=================="
	@if [ -d ".venv" ]; then echo "✅ Virtual environment: .venv exists"; else echo "❌ Virtual environment: .venv missing"; fi
	@if [ -f ".env" ]; then echo "✅ Environment file: .env exists"; else echo "⚠️  Environment file: .env missing (copy from .env.dist)"; fi
	@if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then echo "✅ Git repository: initialized"; else echo "⚠️  Git repository: not initialized"; fi
	@echo ""
	@echo "📦 Package info:"
	@if [ -f ".venv/bin/python" ]; then $(PYTHON) -c "import llm_libration; print(f'Version: {llm_libration.__version__}')"; fi

setup: ## Complete project setup for new users
	@echo "🚀 Setting up llm-libration development environment..."
	@$(MAKE) install
	@if [ ! -f ".env" ]; then echo "📝 Creating .env file from template..."; cp .env.dist .env; echo "⚠️  Please edit .env and add your OpenAI API key"; fi
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env and add your OpenAI API key"
	@echo "2. Activate environment: source .venv/bin/activate"
	@echo "3. Run tests: make test" 