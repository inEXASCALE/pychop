"""
Test script for JAX backend quantization: PTQ and QAT.

This script demonstrates:
1. Building a simple CNN with Flax
2. Training in full precision
3. Post-training quantization (PTQ)
4. Quantization-aware training (QAT)
5. Comparing accuracy across different quantization methods
"""

import os
os.environ["chop_backend"] = "jax"  # Set backend before importing pychop

import jax
import jax.numpy as jnp
from jax import random, jit, grad
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from typing import Any, Callable
import time

# Import pychop components
import pychop
from pychop.jx.layers import (
    ChopSTE, 
    QuantizedConv2d, 
    QuantizedDense,
    QuantizedReLU,
    QuantizedBatchNorm2d,
    post_quantization
)


# ===================================================================
# 1. Create Synthetic MNIST-like Dataset
# ===================================================================

def create_synthetic_dataset(num_samples=1000, num_classes=10, image_size=28):
    """Create a synthetic dataset for testing.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    num_classes : int
        Number of classes.
    image_size : int
        Height/width of square images.
    
    Returns
    -------
    images : jnp.ndarray
        Images of shape (num_samples, image_size, image_size, 1).
    labels : jnp.ndarray
        Labels of shape (num_samples,).
    """
    key = random.PRNGKey(42)
    
    # Generate random images
    images = random.normal(key, (num_samples, image_size, image_size, 1))
    
    # Generate random labels
    labels = random.randint(key, (num_samples,), 0, num_classes)
    
    return images, labels


def create_batches(images, labels, batch_size=32):
    """Create batches from dataset.
    
    Parameters
    ----------
    images : jnp.ndarray
        Images array.
    labels : jnp.ndarray
        Labels array.
    batch_size : int
        Batch size.
    
    Yields
    ------
    tuple
        (batch_images, batch_labels)
    """
    num_samples = images.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield images[batch_indices], labels[batch_indices]


# ===================================================================
# 2. Define Full-Precision Model (Flax)
# ===================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for classification (full precision).
    
    Architecture:
    - Conv2d(1 -> 16, 3x3) -> ReLU -> BatchNorm
    - Conv2d(16 -> 32, 3x3) -> ReLU -> BatchNorm
    - Flatten
    - Dense(32*28*28 -> 128) -> ReLU
    - Dense(128 -> 10)
    
    Attributes
    ----------
    num_classes : int
        Number of output classes.
    """
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """Forward pass.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input images of shape (batch, height, width, channels).
        training : bool
            Whether in training mode (for batch norm).
        
        Returns
        -------
        jnp.ndarray
            Logits of shape (batch, num_classes).
        """
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = x.reshape((x.shape[0], -1))  # Flatten
        
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=self.num_classes)(x)
        
        return x


# ===================================================================
# 3. Define Quantized Model for QAT
# ===================================================================

class QuantizedCNN(nn.Module):
    """Quantized CNN for QAT.
    
    Same architecture as SimpleCNN but with quantized layers.
    
    Attributes
    ----------
    num_classes : int
        Number of output classes.
    chop : ChopSTE or None
        Quantizer instance with STE.
    """
    num_classes: int = 10
    chop: Any = None
    
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
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # FC layers
        x = QuantizedDense(features=128, chop=self.chop)(x)
        x = QuantizedReLU(chop=self.chop)(x)
        
        x = QuantizedDense(features=self.num_classes, chop=self.chop)(x)
        
        return x


# ===================================================================
# 4. Training Utilities
# ===================================================================

def create_train_state(model, rng, learning_rate, input_shape):
    """Create initial training state.
    
    Parameters
    ----------
    model : nn.Module
        Flax model.
    rng : jax.random.PRNGKey
        Random key for initialization.
    learning_rate : float
        Learning rate.
    input_shape : tuple
        Shape of input (batch, height, width, channels).
    
    Returns
    -------
    train_state.TrainState
        Initial training state.
    """
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input, training=False)
    
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    ), variables.get('batch_stats', None)


@jit
def compute_loss(logits, labels):
    """Compute cross-entropy loss.
    
    Parameters
    ----------
    logits : jnp.ndarray
        Model predictions of shape (batch, num_classes).
    labels : jnp.ndarray
        Ground truth labels of shape (batch,).
    
    Returns
    -------
    float
        Average cross-entropy loss.
    """
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(loss)


@jit
def compute_accuracy(logits, labels):
    """Compute classification accuracy.
    
    Parameters
    ----------
    logits : jnp.ndarray
        Model predictions.
    labels : jnp.ndarray
        Ground truth labels.
    
    Returns
    -------
    float
        Accuracy (fraction of correct predictions).
    """
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


def train_step(state, batch_stats, batch, model):
    """Perform one training step.
    
    Parameters
    ----------
    state : train_state.TrainState
        Current training state.
    batch_stats : dict or None
        Batch normalization statistics.
    batch : tuple
        (images, labels).
    model : nn.Module
        Flax model.
    
    Returns
    -------
    tuple
        (new_state, new_batch_stats, loss, accuracy)
    """
    images, labels = batch
    
    def loss_fn(params):
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits, new_model_state = model.apply(
                variables, images, training=True, mutable=['batch_stats']
            )
            return compute_loss(logits, labels), (logits, new_model_state)
        else:
            logits = model.apply({'params': params}, images, training=True)
            return compute_loss(logits, labels), (logits, None)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    new_batch_stats = None
    if new_model_state is not None:
        new_batch_stats = new_model_state['batch_stats']
    
    accuracy = compute_accuracy(logits, labels)
    
    return state, new_batch_stats, loss, accuracy


def eval_model(state, batch_stats, images, labels, model):
    """Evaluate model on a dataset.
    
    Parameters
    ----------
    state : train_state.TrainState
        Training state.
    batch_stats : dict or None
        Batch normalization statistics.
    images : jnp.ndarray
        Images to evaluate.
    labels : jnp.ndarray
        Ground truth labels.
    model : nn.Module
        Flax model.
    
    Returns
    -------
    tuple
        (loss, accuracy)
    """
    if batch_stats is not None:
        variables = {'params': state.params, 'batch_stats': batch_stats}
        logits = model.apply(variables, images, training=False)
    else:
        logits = model.apply({'params': state.params}, images, training=False)
    
    loss = compute_loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)
    
    return loss, accuracy


def train_model(model, state, batch_stats, train_images, train_labels, 
                test_images, test_labels, num_epochs=5, batch_size=32):
    """Train a model for multiple epochs.
    
    Parameters
    ----------
    model : nn.Module
        Flax model.
    state : train_state.TrainState
        Initial training state.
    batch_stats : dict or None
        Initial batch statistics.
    train_images : jnp.ndarray
        Training images.
    train_labels : jnp.ndarray
        Training labels.
    test_images : jnp.ndarray
        Test images.
    test_labels : jnp.ndarray
        Test labels.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    
    Returns
    -------
    tuple
        (final_state, final_batch_stats)
    """
    print(f"\n{'='*70}")
    print(f"Training for {num_epochs} epochs...")
    print(f"{'='*70}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch in create_batches(train_images, train_labels, batch_size):
            state, batch_stats, loss, acc = train_step(state, batch_stats, batch, model)
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1
        
        epoch_time = time.time() - start_time
        
        # Compute averages
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # Evaluate on test set
        test_loss, test_acc = eval_model(
            state, batch_stats, test_images, test_labels, model
        )
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")
    
    return state, batch_stats


# ===================================================================
# 5. Main Test Function
# ===================================================================

def main():
    """Main test function for PTQ and QAT."""
    
    print("\n" + "="*70)
    print("JAX Backend Quantization Test: PTQ vs QAT")
    print("="*70)
    
    # Set random seed
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    
    # ---------------------------------------------------------------
    # Step 1: Create dataset
    # ---------------------------------------------------------------
    print("\n[Step 1] Creating synthetic dataset...")
    train_images, train_labels = create_synthetic_dataset(
        num_samples=1000, num_classes=10
    )
    test_images, test_labels = create_synthetic_dataset(
        num_samples=200, num_classes=10
    )
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # ---------------------------------------------------------------
    # Step 2: Train full-precision model
    # ---------------------------------------------------------------
    print("\n[Step 2] Training full-precision model...")
    fp_model = SimpleCNN(num_classes=10)
    fp_state, fp_batch_stats = create_train_state(
        fp_model, init_rng, learning_rate=1e-3, 
        input_shape=(1, 28, 28, 1)
    )
    
    fp_state, fp_batch_stats = train_model(
        fp_model, fp_state, fp_batch_stats,
        train_images, train_labels,
        test_images, test_labels,
        num_epochs=3, batch_size=32
    )
    
    # Final evaluation
    fp_loss, fp_acc = eval_model(
        fp_state, fp_batch_stats, test_images, test_labels, fp_model
    )
    print(f"\n[Full Precision] Final Test Accuracy: {fp_acc:.4f}")
    
    # ---------------------------------------------------------------
    # Step 3: Post-Training Quantization (PTQ)
    # ---------------------------------------------------------------
    print("\n[Step 3] Performing Post-Training Quantization (PTQ)...")
    
    # Create quantizer (FP16: 5 exp bits, 10 sig bits)
    chop_ptq = pychop.Chop(exp_bits=2, sig_bits=2, rmode=1, subnormal=True)
    
    # Quantize parameters
    if fp_batch_stats is not None:
        variables_to_quantize = {
            'params': fp_state.params,
            'batch_stats': fp_batch_stats
        }
    else:
        variables_to_quantize = fp_state.params
    
    print(f"  Quantizing to FP format: exp_bits=5, sig_bits=10")
    quantized_params = post_quantization(
        variables_to_quantize, 
        chop_ptq, 
        eval_mode=True, 
        verbose=True
    )
    
    # Evaluate PTQ model
    if fp_batch_stats is not None:
        ptq_state = fp_state.replace(params=quantized_params['params'])
        ptq_batch_stats = quantized_params['batch_stats']
    else:
        ptq_state = fp_state.replace(params=quantized_params)
        ptq_batch_stats = None
    
    ptq_loss, ptq_acc = eval_model(
        ptq_state, ptq_batch_stats, test_images, test_labels, fp_model
    )
    print(f"\n[PTQ] Test Accuracy: {ptq_acc:.4f}")
    print(f"[PTQ] Accuracy Drop: {fp_acc - ptq_acc:.4f}")
        
    # 在测试代码中添加这个验证
    print("\n[Verification] Checking if quantization was applied...")

    # 检查某个参数的实际值
    original_kernel = fp_state.params['Dense_0']['kernel']
    quantized_kernel = ptq_state.params['Dense_0']['kernel']

    print(f"\nOriginal kernel sample values: {original_kernel.flatten()[:5]}")
    print(f"Quantized kernel sample values: {quantized_kernel.flatten()[:5]}")
    print(f"Max absolute difference: {jnp.max(jnp.abs(original_kernel - quantized_kernel)):.6f}")
    print(f"Mean absolute difference: {jnp.mean(jnp.abs(original_kernel - quantized_kernel)):.6f}")

    # 检查是否所有值都不同
    num_different = jnp.sum(original_kernel != quantized_kernel)
    total_params = original_kernel.size
    print(f"Parameters changed: {num_different}/{total_params} ({100*num_different/total_params:.2f}%)")

    # ---------------------------------------------------------------
    # Step 4: Quantization-Aware Training (QAT)
    # ---------------------------------------------------------------
    print("\n[Step 4] Performing Quantization-Aware Training (QAT)...")
    
    # Create quantizer with STE
    chop_qat = ChopSTE(exp_bits=5, sig_bits=10, rmode=1, subnormal=True)
    print(f"  Using ChopSTE with exp_bits=5, sig_bits=10")
    
    # Create quantized model
    qat_model = QuantizedCNN(num_classes=10, chop=chop_qat)
    
    # Initialize from scratch or from pre-trained weights
    print("\n  Option A: Training QAT model from scratch...")
    qat_state, qat_batch_stats = create_train_state(
        qat_model, init_rng, learning_rate=1e-3,
        input_shape=(1, 28, 28, 1)
    )
    
    qat_state, qat_batch_stats = train_model(
        qat_model, qat_state, qat_batch_stats,
        train_images, train_labels,
        test_images, test_labels,
        num_epochs=5, batch_size=32
    )
    
    # Final evaluation
    qat_loss, qat_acc = eval_model(
        qat_state, qat_batch_stats, test_images, test_labels, qat_model
    )
    print(f"\n[QAT] Final Test Accuracy: {qat_acc:.4f}")
    
    # ---------------------------------------------------------------
    # Step 5: Compare Results
    # ---------------------------------------------------------------
    print("\n" + "="*70)
    print("Final Comparison")
    print("="*70)
    print(f"Full Precision:    Accuracy = {fp_acc:.4f}")
    print(f"PTQ (FP16):        Accuracy = {ptq_acc:.4f}  (Drop: {fp_acc - ptq_acc:+.4f})")
    print(f"QAT (FP16):        Accuracy = {qat_acc:.4f}  (Drop: {fp_acc - qat_acc:+.4f})")
    print("="*70)
    
    if qat_acc > ptq_acc:
        improvement = qat_acc - ptq_acc
        print(f"\n✅ QAT improves over PTQ by {improvement:.4f} accuracy points!")
    else:
        print(f"\n⚠️  PTQ performs better (this can happen with synthetic data)")
    
    print("\n" + "="*70)
    print("Test completed successfully! 🎉")
    print("="*70)


# ===================================================================
# 6. Run Tests
# ===================================================================

if __name__ == "__main__":
    # Check JAX availability
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("Error: JAX is not installed. Install with: pip install jax jaxlib")
        exit(1)
    
    # Check Flax availability
    try:
        from flax import linen as nn
        import flax
        print(f"Flax version: {flax.__version__}")
    except ImportError:
        print("Error: Flax is not installed. Install with: pip install flax")
        exit(1)
    
    # Check pychop
    try:
        import pychop
        print(f"PyChop backend: {pychop.backend()}")
    except ImportError:
        print("Error: pychop is not installed.")
        exit(1)
    
    # Run main test
    main()