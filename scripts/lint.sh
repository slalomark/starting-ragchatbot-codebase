#!/bin/bash
#
# Run code quality checks
#

set -e

echo "🔍 Running code quality checks..."

echo "📋 Checking code with ruff..."
uv run ruff check backend/

echo "🎨 Checking code formatting with black..."
uv run black --check backend/

echo "✅ All code quality checks passed!"