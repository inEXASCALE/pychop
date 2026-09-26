import importlib.util
import json
from pathlib import Path
import numpy as np


def test_toy_models_export(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples" / "p3109" / "toy_models.py"
    spec = importlib.util.spec_from_file_location("p3109_toys", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.run(tmp_path)
    assert set(metrics) == {"sine", "ar1", "oscillator", "symbol_classifier"}
    for name in ["sine", "ar1", "oscillator"]:
        assert 0 <= metrics[name]["quantization_rmse"] < .05
        assert np.isfinite(metrics[name]["symbol_reconstruction_rmse"])
    assert (tmp_path / "parameters.json").exists()
    with np.load(tmp_path / "results.npz", allow_pickle=False) as output:
        assert output["symbols"].shape == (3, 16)
        assert output["float_codes"].dtype == np.uint8
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
