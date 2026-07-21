#!/usr/bin/env bash
# Remove Python/build/test caches, then reformat the repo (isort + black).
# Needs the dev extras: pip install -e .[dev]  (or: pip install isort black)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "========================================"
echo "Cleaning Python cache files..."
echo "========================================"

# Remove __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove compiled python files
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete

# Remove pytest cache
find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Remove mypy cache
find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Remove Ruff cache
find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Remove Pyre cache
find . -type d -name ".pyre" -exec rm -rf {} +

# Remove Pytype cache
find . -type d -name ".pytype" -exec rm -rf {} +

# Remove tox / nox
find . -type d -name ".tox" -exec rm -rf {} +
find . -type d -name ".nox" -exec rm -rf {} +

# Remove coverage files
find . -type f -name ".coverage*" -delete
find . -type d -name "htmlcov" -exec rm -rf {} +

# Remove build artifacts
rm -rf build dist
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type f -name "*.egg" -delete

echo
echo "========================================"
echo "Running isort..."
echo "========================================"

isort .

echo
echo "========================================"
echo "Running black..."
echo "========================================"

black .

echo
echo "========================================"
echo "Done!"
echo "========================================"
