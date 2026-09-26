"""Train a small P3109-quantized linear model on a selected native backend.

python examples/p3109/tensor_training.py --backend torch --output-dir /tmp/p3109-torch
Choose tensorflow or jax for the other native examples. No dataset is downloaded.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pychop import P3109, P3109Format


def run(backend, output_dir):
    rng = np.random.default_rng(19)
    data = rng.normal(size=(64, 3)).astype(np.float32)
    target = data @ np.array([[.75], [-.5], [.25]], dtype=np.float32)
    q = P3109(P3109Format(8, 4), saturate=True)
    if backend == "torch":
        import torch
        x, y = torch.tensor(data), torch.tensor(target)
        weights = torch.zeros((3, 1), requires_grad=True)
        losses = []
        for _ in range(25):
            prediction = q(q(x) @ q(weights, ste=True), ste=True)
            loss = ((prediction - y)**2).mean()
            loss.backward()
            with torch.no_grad():
                weights -= .15 * weights.grad
                weights.grad.zero_()
            losses.append(float(loss.detach()))
        codes = q.encode(weights)
        restored = q.decode(codes, dtype="float32")
        np.testing.assert_array_equal(restored.numpy(), q(weights).numpy())
        code_array = codes.numpy()
    elif backend == "tensorflow":
        import tensorflow as tf
        x, y = tf.constant(data), tf.constant(target)
        weights = tf.Variable(tf.zeros((3, 1)))

        @tf.function(autograph=False)
        def step():
            with tf.GradientTape() as tape:
                prediction = q(tf.matmul(q(x), q(weights, ste=True)), ste=True)
                loss = tf.reduce_mean((prediction - y)**2)
            weights.assign_sub(.15 * tape.gradient(loss, weights))
            return loss

        losses = [float(step()) for _ in range(25)]
        codes = q.encode(weights)
        np.testing.assert_array_equal(q.decode(codes, dtype="float32").numpy(), q(weights).numpy())
        code_array = codes.numpy()
    elif backend == "jax":
        import jax
        import jax.numpy as jnp
        x, y = jnp.asarray(data), jnp.asarray(target)
        weights = jnp.zeros((3, 1), dtype=jnp.float32)

        @jax.jit
        @jax.value_and_grad
        def objective(w):
            prediction = q(q(x) @ q(w, ste=True), ste=True)
            return jnp.mean((prediction - y)**2)

        losses = []
        for _ in range(25):
            loss, gradient = objective(weights)
            weights = weights - .15 * gradient
            losses.append(float(loss))
        codes = q.encode(weights)
        np.testing.assert_array_equal(np.asarray(q.decode(codes, dtype="float32")), np.asarray(q(weights)))
        code_array = np.asarray(codes)
    else:
        raise ValueError("backend must be torch, tensorflow or jax")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy.json").write_text(json.dumps(q.to_dict(), indent=2) + "\n")
    np.savez_compressed(output_dir / "model.npz", weight_codes=code_array, losses=losses)
    restored_policy = P3109.from_dict(json.loads((output_dir / "policy.json").read_text()))
    # Codes are portable: the same exported weights can be decoded with NumPy.
    prediction = restored_policy(restored_policy(data) @ restored_policy.decode(code_array))
    return {"backend": backend, "initial_loss": losses[0], "final_training_loss": losses[-1],
            "exported_model_mse": float(np.mean((prediction - target)**2))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "tensorflow", "jax"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.backend, args.output_dir), indent=2))
