#!bash
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

PATTERN='Copyright \(c\) 202[0-9] Graphcore Ltd\. +All rights reserved\.'

# We "grep ." so the exit code signals that the first grep generated output
if grep -L -E "$PATTERN" "$@" | grep .
then
  # There was output, signal unsuccessful
  exit 1
fi
# Normal exit, signalling success
