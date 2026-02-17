# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

tmpdir=$(mktemp -d)
test -d $tmpdir || exit -1

cleanup () {
  echo "Removing $tmpdir"
  rm $tmpdir/t.sh
  rmdir $tmpdir
}

trap cleanup EXIT

# Passing case
echo "Copyright (c) 2024 Graphcore Ltd. All rights reserved." > $tmpdir/t.sh
if sh etc/check-copyright.sh $tmpdir/t.sh
then
  echo Pass: Should have passed
else
  echo FAIL: Should have passed
fi

# Failing case
echo "Copyright (c) 2024 Graphcore Ltd. All rights xreserved." > $tmpdir/t.sh
if sh etc/check-copyright.sh $tmpdir/t.sh
then
  echo FAIL: Should have failed, but passed
else
  echo Pass: Should have failed
fi
