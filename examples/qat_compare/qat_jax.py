"""
Example 4: PyChop + JAX/Flax (Quantized FP16)

This script demonstrates quantization-aware training (QAT) using PyChop with
JAX/Flax backend. It shows how to use custom floating-point precision (FP16)
for training and deployment.

MNIST digit classification
Framework: JAX/Flax + PyChop
Quantization: FP16 (5 exp bits, 10 sig bits) with QAT
"""

import os
os.environ["chop_backend"] = "jax"  # Set JAX backend

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax import linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
import numpy as np
import time
import pickle

# Import PyChop
import pychop
from pychop.jx.layers import (
    ChopSTE,
    QuantizedConv2d,
    QuantizedDense,
    QuantizedReLU,
    QuantizedBatchNorm2d,
    QuantizedDropout,
    post_quantization
)


# ===================================================================
# 1. Define Quantized Model Architecture (PyChop + JAX)
# ===================================================================

class QuantizedMNISTNet(nn.Module):
    """Quantized CNN for MNIST classification using PyChop.
    
    Architecture:
    - QuantizedConv2d(1->16, 3x3) -> QuantizedReLU -> QuantizedBatchNorm -> MaxPool(2x2)
    - QuantizedConv2d(16->32, 3x3) -> QuantizedReLU -> QuantizedBatchNorm -> MaxPool(2x2)
    - Flatten
    - QuantizedDense(32*7*7 -> 128) -> QuantizedReLU -> QuantizedDropout(0.5)
    - QuantizedDense(128 -> 10)
    
    Attributes
    ----------
    num_classes : int
        Number of output classes.
    chop : ChopSTE
        Quantizer instance with straight-through estimator.
    """
    
    num_classes: int = 10
    chop: any = None
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """Forward pass with quantization.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input images of shape (batch, height, width, channels).
        training : bool
            Whether in training mode.
        
        Returns
        -------
        jnp.ndarray
            Logits of shape (batch, num_classes).
        """
        # Layer 1
        x = QuantizedConv2d(
            features=16, 
            kernel_size=(3, 3), 
            padding='SAME',
            chop=self.chop
        )(x)
        x = QuantizedReLU(chop=self.chop)(x)
        x = QuantizedBatchNorm2d(
            use_running_average=not training,
            chop=self.chop
        )(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Layer 2
        x = QuantizedConv2d(
            features=32, 
            kernel_size=(3, 3), 
            padding='SAME',
            chop=self.chop
        )(x)
        x = QuantizedReLU(chop=self.chop)(x)
        x = QuantizedBatchNorm2d(
            use_running_average=not training,
            chop=self.chop
        )(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # FC layers
        x = QuantizedDense(features=128, chop=self.chop)(x)
        x = QuantizedReLU(chop=self.chop)(x)
        x = QuantizedDropout(
            rate=0.5, 
            deterministic=not training,
            chop=self.chop
        )(x)
        x = QuantizedDense(features=self.num_classes, chop=self.chop)(x)
        
        return x


# Standard model for PTQ comparison
class MNISTNet(nn.Module):
    """Standard CNN for PTQ comparison."""
    
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x


# ===================================================================
# 2. Data Loading (FIXED)
# ===================================================================

def load_mnist_data():
    """Load MNIST dataset using TensorFlow Datasets.
    
    Returns
    -------
    tuple
        (train_images, train_labels), (test_images, test_labels)
    """
    def prepare_data(dataset):
        """Convert TensorFlow dataset to NumPy arrays."""
        images = []
        labels = []
        
        for example in dataset:
            # Convert TensorFlow tensor to NumPy
            img = example['image'].numpy().astype(np.float32) / 255.0
            lbl = example['label'].numpy()
            images.append(img)
            labels.append(lbl)
        
        return np.array(images), np.array(labels)
    
    print("  Downloading MNIST dataset...")
    
    # Load dataset
    train_ds = tfds.load('mnist', split='train', as_supervised=False)
    test_ds = tfds.load('mnist', split='test', as_supervised=False)
    
    print("  Converting to NumPy arrays...")
    train_images, train_labels = prepare_data(train_ds)
    test_images, test_labels = prepare_data(test_ds)
    
    return (train_images, train_labels), (test_images, test_labels)


def create_batches(images, labels, batch_size, rng):
    """Create shuffled batches."""
    num_samples = len(images)
    indices = jax.random.permutation(rng, num_samples)
    indices = np.array(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield images[batch_indices], labels[batch_indices]


# ===================================================================
# 3. Training Functions (Same as before)
# ===================================================================

def create_train_state(model, rng, learning_rate, input_shape):
    """Create initial training state."""
    variables = model.init(rng, jnp.ones(input_shape), training=False)
    tx = optax.adam(learning_rate)
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    batch_stats = variables.get('batch_stats', None)
    
    return state, batch_stats


@jax.jit
def train_step(state, batch_stats, images, labels, dropout_rng):
    """Perform one training step."""
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': batch_stats}
        logits, new_model_state = state.apply_fn(
            variables, 
            images, 
            training=True, 
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        
        one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
        loss = jnp.mean(loss)
        
        return loss, (logits, new_model_state)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    new_batch_stats = new_model_state['batch_stats']
    
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    
    return state, new_batch_stats, loss, accuracy


@jax.jit
def eval_step(state, batch_stats, images, labels):
    """Perform one evaluation step."""
    variables = {'params': state.params, 'batch_stats': batch_stats}
    logits = state.apply_fn(variables, images, training=False)
    
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    loss = jnp.mean(loss)
    
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    
    return loss, accuracy


def evaluate_model(state, batch_stats, test_images, test_labels, batch_size=128):
    """Evaluate model on entire test set."""
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for start_idx in range(0, len(test_images), batch_size):
        end_idx = min(start_idx + batch_size, len(test_images))
        batch_images = test_images[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        
        loss, acc = eval_step(state, batch_stats, batch_images, batch_labels)
        total_loss += loss
        total_acc += acc
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return float(avg_loss), float(avg_acc)


def count_parameters_flax(params):
    """Count total parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def get_model_size_flax(params, bits_per_param=32):
    """Get model size in MB."""
    num_params = count_parameters_flax(params)
    size_mb = num_params * bits_per_param / 8 / 1024 / 1024
    return size_mb


# ===================================================================
# 4. Main Training Pipeline
# ===================================================================

def main():
    """Main QAT training pipeline with PyChop."""
    
    print("="*70)
    print("Example 4: PyChop + JAX/Flax (Quantized FP16 with QAT)")
    print("="*70)
    
    # Set random seed
    rng = random.PRNGKey(42)
    
    print(f"\nJAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    # ---------------------------------------------------------------
    # Load MNIST Dataset
    # ---------------------------------------------------------------
    print("\n[Step 1] Loading MNIST dataset...")
    
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
    
    print(f"  Train samples: {len(train_images)}")
    print(f"  Test samples: {len(test_images)}")
    print(f"  Image shape: {train_images.shape[1:]}")
    
    # ---------------------------------------------------------------
    # Create Quantizer
    # ---------------------------------------------------------------
    print("\n[Step 2] Creating quantizer...")
    
    # FP16: 5 exponent bits, 10 significand bits (IEEE 754 half precision)
    chop = ChopSTE(exp_bits=5, sig_bits=10, rmode=1, subnormal=True)
    
    print(f"  Quantization format: FP16")
    print(f"  Exponent bits: 5")
    print(f"  Significand bits: 10")
    print(f"  Total bits: 16 (including sign bit)")
    print(f"  Unit roundoff (u): {chop.u:.6e}")
    
    # ---------------------------------------------------------------
    # Create Quantized Model for QAT
    # ---------------------------------------------------------------
    print("\n[Step 3] Creating quantized model for QAT...")
    
    model = QuantizedMNISTNet(num_classes=10, chop=chop)
    
    rng, init_rng = random.split(rng)
    state, batch_stats = create_train_state(
        model, 
        init_rng, 
        learning_rate=0.001,
        input_shape=(1, 28, 28, 1)
    )
    
    num_params = count_parameters_flax(state.params)
    model_size_fp32 = get_model_size_flax(state.params, bits_per_param=32)
    model_size_fp16 = get_model_size_flax(state.params, bits_per_param=16)
    
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size (FP32): {model_size_fp32:.2f} MB")
    print(f"  Model size (FP16): {model_size_fp16:.2f} MB")
    print(f"  Size reduction: {(1 - model_size_fp16/model_size_fp32)*100:.1f}%")
    
    # ---------------------------------------------------------------
    # Training Configuration
    # ---------------------------------------------------------------
    print("\n[Step 4] Training configuration...")
    
    num_epochs = 5
    batch_size = 128
    
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training mode: Quantization-Aware Training (QAT)")
    
    # ---------------------------------------------------------------
    # Train Model with QAT
    # ---------------------------------------------------------------
    print("\n[Step 5] Training with Quantization-Aware Training...")
    print("="*70)
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        # Create batches
        rng, shuffle_rng, dropout_rng = random.split(rng, 3)
        
        for batch_images, batch_labels in create_batches(
            train_images, train_labels, batch_size, shuffle_rng
        ):
            dropout_rng, step_rng = random.split(dropout_rng)
            
            state, batch_stats, loss, acc = train_step(
                state, batch_stats, batch_images, batch_labels, step_rng
            )
            
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(
            state, batch_stats, test_images, test_labels, batch_size
        )
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc*100:.2f}%, Time={epoch_time:.2f}s")
        print(f"  Test: Loss={test_loss:.4f}, Acc={test_acc*100:.2f}%")
        print("-"*70)
    
    # ---------------------------------------------------------------
    # Final Evaluation
    # ---------------------------------------------------------------
    print("\n[Step 6] Final evaluation...")
    print("="*70)
    
    test_loss, test_acc = evaluate_model(
        state, batch_stats, test_images, test_labels, batch_size
    )
    
    print(f"\nFinal Results (PyChop + JAX/Flax FP16 QAT):")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Model Size (deployed): {model_size_fp16:.2f} MB")
    print(f"  Precision: 16-bit (FP16)")
    print(f"  Training method: Quantization-Aware Training (QAT)")
    
    # ---------------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------------
    print("\n[Step 7] Saving quantized model...")
    
    checkpoint = {
        'params': state.params,
        'batch_stats': batch_stats,
        'test_accuracy': float(test_acc),
        'num_parameters': num_params,
        'quantization': {
            'exp_bits': 5,
            'sig_bits': 10,
            'format': 'FP16',
            'method': 'QAT'
        }
    }
    
    with open('mnist_pychop_jax_fp16_qat.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print("  Model saved to: mnist_pychop_jax_fp16_qat.pkl")
    
    # ---------------------------------------------------------------
    # Optional: Post-Training Quantization for Comparison
    # ---------------------------------------------------------------
    print("\n[Step 8] Bonus: PTQ from full-precision model...")
    
    # Train a full-precision model first
    print("  Training full-precision model...")
    fp_model = MNISTNet(num_classes=10)
    
    rng, init_rng = random.split(rng)
    fp_state, fp_batch_stats = create_train_state(
        fp_model,
        init_rng,
        learning_rate=0.001,
        input_shape=(1, 28, 28, 1)
    )
    
    # Quick training (2 epochs)
    for epoch in range(1, 3):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        rng, shuffle_rng, dropout_rng = random.split(rng, 3)
        
        for batch_images, batch_labels in create_batches(
            train_images, train_labels, batch_size, shuffle_rng
        ):
            dropout_rng, step_rng = random.split(dropout_rng)
            
            fp_state, fp_batch_stats, loss, acc = train_step(
                fp_state, fp_batch_stats, batch_images, batch_labels, step_rng
            )
            
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1
        
        print(f"  Epoch {epoch}: Loss={epoch_loss/num_batches:.4f}, Acc={epoch_acc/num_batches*100:.2f}%")
    
    fp_loss, fp_acc = evaluate_model(
        fp_state, fp_batch_stats, test_images, test_labels, batch_size
    )
    print(f"  Full-precision accuracy: {fp_acc*100:.2f}%")
    
    # Apply PTQ
    print("\n  Applying post-training quantization...")
    chop_ptq = ChopSTE(exp_bits=5, sig_bits=10, rmode=1, subnormal=True)
    
    variables_to_quantize = {
        'params': fp_state.params,
        'batch_stats': fp_batch_stats
    }
    
    quantized_vars = post_quantization(
        variables_to_quantize, 
        chop_ptq, 
        eval_mode=True, 
        verbose=False
    )
    
    ptq_state = fp_state.replace(params=quantized_vars['params'])
    ptq_batch_stats = quantized_vars['batch_stats']
    
    ptq_loss, ptq_acc = evaluate_model(
        ptq_state, ptq_batch_stats, test_images, test_labels, batch_size
    )
    print(f"  PTQ accuracy: {ptq_acc*100:.2f}%")
    print(f"  Accuracy drop (PTQ): {(fp_acc - ptq_acc)*100:.2f}%")
    
    print("\n" + "="*70)
    print("Comparison: QAT vs PTQ")
    print("="*70)
    print(f"QAT (trained with quantization): {test_acc*100:.2f}%")
    print(f"PTQ (quantized after training):  {ptq_acc*100:.2f}%")
    print(f"Full Precision (baseline):       {fp_acc*100:.2f}%")
    print("="*70)
    
    print("\n" + "="*70)
    print("Training completed successfully! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()