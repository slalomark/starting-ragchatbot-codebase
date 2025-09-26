#!/bin/bash
#
# Run full quality checks: linting, formatting, and tests
#

set -e

echo "🚀 Running comprehensive quality checks..."

echo "🔍 Step 1: Code quality checks..."
./scripts/lint.sh

echo "🧪 Step 2: Running tests..."
uv run pytest backend/tests/ -v --tb=short

echo "📊 Step 3: Running tests with coverage..."
uv run pytest backend/tests/ --cov=backend --cov-report=term-missing

echo "✅ All quality checks passed! Code is ready for commit."