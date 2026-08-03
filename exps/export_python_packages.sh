#!/usr/bin/env bash

###############################################################################
# Script Name:
#   export_python_packages.sh
#
# Description:
#   Export all installed Python packages and their exact versions from the
#   current Python environment into a text file.
#
#   The output format is compatible with "pip install -r", for example:
#
#       numpy==2.1.3
#       pandas==2.2.3
#       requests==2.32.3
#
#   This script uses:
#
#       python -m pip
#
#   instead of calling "pip" directly. This is important because it ensures
#   that the pip command belongs to the same Python interpreter that is
#   currently being used.
#
# Usage:
#
#   1. Export packages using the default Python:
#
#       ./export_python_packages.sh
#
#      Default output file:
#
#       python_packages.txt
#
#   2. Specify a custom output file:
#
#       ./export_python_packages.sh requirements_backup.txt
#
#   3. Specify a different Python executable:
#
#       PYTHON_BIN=python3.11 ./export_python_packages.sh
#
#   4. Export packages from an activated virtual environment:
#
#       source .venv/bin/activate
#       ./export_python_packages.sh
#
# Example output:
#
#       Python executable : /path/to/.venv/bin/python
#       Python version    : Python 3.11.9
#       Output file       : python_packages.txt
#
# Notes:
#
#   - It is recommended to activate the desired virtual environment before
#     running this script.
#
#   - The generated file can be used by:
#
#         install_python_packages.sh
#
#   - Package versions are exported in "package==version" format.
#
#   - Local/editable packages or packages installed from special sources may
#     require additional handling depending on how they were installed.
#
###############################################################################

set -euo pipefail


###############################################################################
# Configuration
###############################################################################

# Python executable.
# Can be overridden from the command line:
#
#   PYTHON_BIN=python3.11 ./export_python_packages.sh
#
PYTHON_BIN="${PYTHON_BIN:-python}"

# Output filename.
# The first command-line argument overrides the default.
#
# Example:
#
#   ./export_python_packages.sh my_packages.txt
#
OUTPUT_FILE="${1:-python_packages.txt}"


###############################################################################
# Check Python
###############################################################################

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi


###############################################################################
# Check pip
###############################################################################

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    echo "ERROR: pip is not available for: $PYTHON_BIN" >&2
    echo "Try installing pip first." >&2
    exit 1
fi


###############################################################################
# Display environment information
###############################################################################

PYTHON_PATH="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1)"

echo "============================================================"
echo " Export Python Packages"
echo "============================================================"
echo
echo "Python executable : $PYTHON_PATH"
echo "Python version    : $PYTHON_VERSION"
echo "Output file       : $OUTPUT_FILE"
echo


###############################################################################
# Export installed packages
###############################################################################

echo "Exporting installed Python packages..."

# "pip list --format=freeze" produces:
#
#   package==version
#
# This format can be directly consumed by:
#
#   python -m pip install -r FILE
#
"$PYTHON_BIN" -m pip list --format=freeze > "$OUTPUT_FILE"


###############################################################################
# Validate output
###############################################################################

if [[ ! -s "$OUTPUT_FILE" ]]; then
    echo "WARNING: Output file is empty: $OUTPUT_FILE" >&2
else
    PACKAGE_COUNT="$(grep -c '==' "$OUTPUT_FILE" || true)"

    echo
    echo "Export completed successfully."
    echo "Packages exported : $PACKAGE_COUNT"
    echo "Saved to          : $OUTPUT_FILE"
fi


###############################################################################
# Show preview
###############################################################################

echo
echo "First 10 packages:"
echo "------------------------------------------------------------"
head -n 10 "$OUTPUT_FILE" || true
echo "------------------------------------------------------------"
echo
echo "Done."