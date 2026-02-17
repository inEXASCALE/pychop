# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

# Set version numbers, make package, and publish

set -o errexit

# This is the master location at which to change version number
VERSION="0.5.2"

# Run the script to change the version elsewhere
perl -pi -e 's/^(release|version) = "([\d.]+)"/$1 = "'$VERSION'"/' docs/source/conf.py
perl -pi -e 's/^version = "([\d.]+)"/version = "'$VERSION'"/' pyproject.toml

# Build docs to embed version
( cd docs && make html )

# Build distribution
rm -rf dist
uv pip install build twine
python -m build
echo "Enter PyPI API Token"
echo __token__ | twine upload --repository pypi dist/* --verbose
