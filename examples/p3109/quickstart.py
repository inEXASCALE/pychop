"""Small executable P3109 examples, including explicit stochastic random bits."""
import json
import numpy as np
from pychop import P3109, P3109Format


def main():
    q = P3109(P3109Format(k=8, precision=4))
    x = np.array([0, .1, 1.0625, 2, 224, 240, np.inf, np.nan])
    print("rounded:", q(x))
    print("codes:", q.encode(x))
    np.testing.assert_array_equal(q.decode(q.encode(x)), q(x))
    print("parameters:", json.dumps(q.to_dict()))
    sr = P3109(q.format, rounding="stochastic_c")
    bits = np.random.default_rng(17).integers(0, 256, size=x.shape, dtype=np.uint32)
    print("stochastic:", sr(x, srbits=bits, srnumbits=8))
    # A matrix multiplication uses host accumulation, then one output rounding.
    weights = q([[1.1, -.3], [.25, .9]])
    print("linear model:", q(weights @ q([.7, -.2])))


if __name__ == "__main__":
    main()
