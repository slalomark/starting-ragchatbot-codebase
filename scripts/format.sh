#!/bin/bash
#
# Format Python code using black and ruff
#

set -e

echo "🎨 Formatting Python code with black..."
uv run black backend/

echo "📝 Sorting imports and fixing auto-fixable issues with ruff..."
uv run ruff check backend/ --fix

echo "✅ Code formatting complete!"