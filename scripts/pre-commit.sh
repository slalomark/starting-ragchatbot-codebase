#!/bin/bash
#
# Pre-commit script: format code and run quality checks
#

set -e

echo "🔧 Pre-commit checks..."

echo "🎨 Step 1: Auto-formatting code..."
./scripts/format.sh

echo "🔍 Step 2: Running quality checks..."
uv run ruff check backend/
uv run black --check backend/

echo "🧪 Step 3: Running tests..."
uv run pytest backend/tests/ -v --tb=short

echo "✅ Pre-commit checks passed! Ready to commit."