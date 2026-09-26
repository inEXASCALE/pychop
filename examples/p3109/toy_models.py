"""Three self-contained models combining P3109 and time-series symbolization.

Run: python examples/p3109/toy_models.py --output-dir /tmp/pychop-demo
No datasets, plotting packages, network, or ML frameworks are required.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pychop import P3109, P3109Format, SymbolicTimeSeries


def signals(n=256):
    """Sine, autoregressive process, and a damped oscillator."""
    rng = np.random.default_rng(2026)
    t = np.arange(n) * .05
    sine = np.sin(2 * np.pi * .4 * t) + .03 * rng.normal(size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = .88 * ar[i - 1] + .12 * rng.normal()
    oscillator = np.exp(-.08 * t) * np.cos(1.8 * t)
    return np.stack([sine, ar, oscillator])


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = signals()
    # Time split precedes fitting: test statistics never enter calibration.
    train, test = data[:, :192], data[:, 192:]
    q = P3109(P3109Format(8, 4), saturate=True)
    symbolizer = SymbolicTimeSeries(segment_size=4, alphabet_size=8).fit(q(train))
    codes = symbolizer.transform(q(test))
    reconstruction = symbolizer.inverse_transform(codes)
    report = {}
    for name, series, reconstructed in zip(["sine", "ar1", "oscillator"], test, reconstruction):
        report[name] = {"quantization_rmse": float(np.sqrt(np.mean((q(series) - series)**2))),
                        "symbol_reconstruction_rmse": float(np.sqrt(np.mean((reconstructed - series)**2)))}

    # Toy model 1: scalar AR(1) least squares, then quantized prediction.
    ar_train, ar_test = train[1], test[1]
    coefficient = float(np.dot(ar_train[:-1], ar_train[1:]) / np.dot(ar_train[:-1], ar_train[:-1]))
    coefficient_q = float(q(coefficient))
    previous = np.concatenate([[ar_train[-1]], ar_test[:-1]])
    prediction = q(coefficient_q * q(previous))
    report["ar1"]["prediction_rmse"] = float(np.sqrt(np.mean((prediction - ar_test)**2)))

    # Toy model 2: explicit-Euler oscillator; each state update is rounded.
    state = np.array([1., 0.])
    reference = state.copy()
    transition = np.array([[1., .05], [-.05 * 1.8**2, 1 - .05 * .16]])
    trajectory = []
    for _ in range(64):
        state = q(q(transition) @ state)
        reference = transition @ reference
        trajectory.append(state.copy())
    report["oscillator"]["final_state_error"] = float(np.linalg.norm(state - reference))

    # Toy model 3: nearest-centroid classification of symbolic sine/oscillator windows.
    features = symbolizer.transform(q(train)).reshape(3, 6, 8)
    centroids = features[[0, 2]].mean(axis=1)
    held_out = codes[[0, 2]].reshape(2, 2, 8)
    labels = np.argmin(((held_out[:, :, None, :] - centroids)**2).sum(axis=-1), axis=-1)
    report["symbol_classifier"] = {"labels": labels.tolist(),
                                  "accuracy": float(np.mean(labels == np.arange(2)[:, None]))}

    # JSON stores policy/calibration/model coefficients; NPZ stores actual code arrays.
    bundle = {"quantizer": q.to_dict(), "symbolizer": symbolizer.to_dict(),
              "model": {"ar1_coefficient": coefficient, "ar1_coefficient_quantized": coefficient_q}}
    (output_dir / "parameters.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output_dir / "results.npz", float_codes=q.encode(test),
                        symbols=codes, reconstruction=reconstruction,
                        prediction=prediction, trajectory=np.array(trajectory))
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    restored = json.loads((output_dir / "parameters.json").read_text())
    q2 = P3109.from_dict(restored["quantizer"])
    s2 = SymbolicTimeSeries.from_dict(restored["symbolizer"])
    with np.load(output_dir / "results.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(q2.decode(saved["float_codes"]), q(test))
        np.testing.assert_array_equal(s2.transform(q2(test)), saved["symbols"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("toy-model-output"))
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
