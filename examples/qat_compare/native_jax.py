"""
Example 2: Raw JAX/Flax Baseline (FP32)

This script demonstrates standard training and deployment of a CNN classifier
using raw JAX/Flax without any quantization. This serves as the JAX baseline.

MNIST digit classification
Framework: JAX/Flax (FP32)
Quantization: None
"""

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


# ===================================================================
# 1. Define Model Architecture (Raw Flax)
# ===================================================================

class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification.
    
    Architecture:
    - Conv(1->16, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)
    - Conv(16->32, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)
    - Flatten
    - Dense(32*7*7 -> 128) -> ReLU -> Dropout(0.5)
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
            Whether in training mode.
        
        Returns
        -------
        jnp.ndarray
            Logits of shape (batch, num_classes).
        """
        # Layer 1
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Layer 2
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # FC layers
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x


# ===================================================================
# 2. Data Loading (Fixed for TensorFlow Tensors)
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
    
    # Load dataset with as_supervised=False to get dict format
    train_ds = tfds.load('mnist', split='train', as_supervised=False)
    test_ds = tfds.load('mnist', split='test', as_supervised=False)
    
    print("  Converting to NumPy arrays...")
    train_images, train_labels = prepare_data(train_ds)
    test_images, test_labels = prepare_data(test_ds)
    
    return (train_images, train_labels), (test_images, test_labels)


def create_batches(images, labels, batch_size, rng):
    """Create shuffled batches.
    
    Parameters
    ----------
    images : np.ndarray
        Images array.
    labels : np.ndarray
        Labels array.
    batch_size : int
        Batch size.
    rng : jax.random.PRNGKey
        Random key for shuffling.
    
    Yields
    ------
    tuple
        (batch_images, batch_labels)
    """
    num_samples = len(images)
    indices = jax.random.permutation(rng, num_samples)
    indices = np.array(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield images[batch_indices], labels[batch_indices]


# ===================================================================
# 3. Training Functions
# ===================================================================

def create_train_state(model, rng, learning_rate, input_shape):
    """Create initial training state.
    
    Parameters
    ----------
    model : nn.Module
        Flax model.
    rng : jax.random.PRNGKey
        Random key.
    learning_rate : float
        Learning rate.
    input_shape : tuple
        Input shape.
    
    Returns
    -------
    tuple
        (train_state, batch_stats)
    """
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
    """Perform one training step.
    
    Parameters
    ----------
    state : train_state.TrainState
        Current training state.
    batch_stats : dict
        Batch normalization statistics.
    images : jnp.ndarray
        Batch images.
    labels : jnp.ndarray
        Batch labels.
    dropout_rng : jax.random.PRNGKey
        Random key for dropout.
    
    Returns
    -------
    tuple
        (new_state, new_batch_stats, loss, accuracy)
    """
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': batch_stats}
        logits, new_model_state = state.apply_fn(
            variables, 
            images, 
            training=True, 
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        
        # Cross-entropy loss
        one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
        loss = jnp.mean(loss)
        
        return loss, (logits, new_model_state)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Update batch stats
    new_batch_stats = new_model_state['batch_stats']
    
    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    
    return state, new_batch_stats, loss, accuracy


@jax.jit
def eval_step(state, batch_stats, images, labels):
    """Perform one evaluation step.
    
    Parameters
    ----------
    state : train_state.TrainState
        Training state.
    batch_stats : dict
        Batch statistics.
    images : jnp.ndarray
        Batch images.
    labels : jnp.ndarray
        Batch labels.
    
    Returns
    -------
    tuple
        (loss, accuracy)
    """
    variables = {'params': state.params, 'batch_stats': batch_stats}
    logits = state.apply_fn(variables, images, training=False)
    
    # Loss
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    loss = jnp.mean(loss)
    
    # Accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    
    return loss, accuracy


def evaluate_model(state, batch_stats, test_images, test_labels, batch_size=128):
    """Evaluate model on entire test set.
    
    Parameters
    ----------
    state : train_state.TrainState
        Training state.
    batch_stats : dict
        Batch statistics.
    test_images : np.ndarray
        Test images.
    test_labels : np.ndarray
        Test labels.
    batch_size : int
        Batch size for evaluation.
    
    Returns
    -------
    tuple
        (average_loss, accuracy)
    """
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
    """Count total parameters in Flax model.
    
    Parameters
    ----------
    params : dict
        Parameter pytree.
    
    Returns
    -------
    int
        Total number of parameters.
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def get_model_size_flax(params):
    """Get model size in MB.
    
    Parameters
    ----------
    params : dict
        Parameter pytree.
    
    Returns
    -------
    float
        Model size in MB (assuming FP32).
    """
    num_params = count_parameters_flax(params)
    size_mb = num_params * 4 / 1024 / 1024  # 4 bytes per FP32
    return size_mb


# ===================================================================
# 4. Main Pipeline
# ===================================================================

def main():
    """Main training and evaluation pipeline."""
    
    print("="*70)
    print("Example 2: Raw JAX/Flax Baseline (FP32)")
    print("="*70)
    
    # Set random seed
    rng = random.PRNGKey(42)
    
    # Device info
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
    # Create Model
    # ---------------------------------------------------------------
    print("\n[Step 2] Creating model...")
    
    model = MNISTNet(num_classes=10)
    
    rng, init_rng = random.split(rng)
    state, batch_stats = create_train_state(
        model, 
        init_rng, 
        learning_rate=0.001,
        input_shape=(1, 28, 28, 1)
    )
    
    num_params = count_parameters_flax(state.params)
    model_size = get_model_size_flax(state.params)
    
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size: {model_size:.2f} MB (FP32)")
    print(f"  Precision: FP32 (32-bit floating point)")
    
    # ---------------------------------------------------------------
    # Training Configuration
    # ---------------------------------------------------------------
    print("\n[Step 3] Training configuration...")
    
    num_epochs = 5
    batch_size = 128
    
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    
    # ---------------------------------------------------------------
    # Train Model
    # ---------------------------------------------------------------
    print("\n[Step 4] Training model...")
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
    print("\n[Step 5] Final evaluation...")
    print("="*70)
    
    test_loss, test_acc = evaluate_model(
        state, batch_stats, test_images, test_labels, batch_size
    )
    
    print(f"\nFinal Results (Raw JAX/Flax FP32):")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Model Size: {model_size:.2f} MB")
    print(f"  Precision: 32-bit")
    
    # ---------------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------------
    print("\n[Step 6] Saving model...")
    
    checkpoint = {
        'params': state.params,
        'batch_stats': batch_stats,
        'test_accuracy': float(test_acc),
        'num_parameters': num_params,
    }
    
    with open('mnist_jax_fp32.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print("  Model saved to: mnist_jax_fp32.pkl")
    
    print("\n" + "="*70)
    print("Training completed successfully! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()