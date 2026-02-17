<!-- Copyright (c) 2024 Graphcore Ltd. All rights reserved. -->

## BUILDING

```
pip install -e .
( cd docs && make html )
# Install packages for testing - will install JAX, Torch, etc.
pip install -r requirements-dev.txt
pytest .
```

#### Pushing
```
sh etc/package.sh
```
