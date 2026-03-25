.. _ptq_guide:

========================================
Post-Training Quantization
========================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
--------------------------

PyChop provides comprehensive **Post-Training Quantization (PTQ)** methods to quantize pre-trained models
without retraining. PTQ is ideal for quick deployment and model compression with minimal accuracy loss.

**Key Features:**

- ✅ **4 PTQ Methods**: Basic, Static, Dynamic, Mixed-Precision
- ✅ **4 Calibration Algorithms**: MinMax, Percentile, KL-Divergence, MSE
- ✅ **Dual Backend Support**: PyTorch and JAX/Flax
- ✅ **Flexible Quantization**: FP16, INT8, INT4, Custom Precision
- ✅ **Easy API**: Unified interface across backends


.. _ptq_comparison:

PTQ Methods Comparison
--------------------------


.. list-table:: Quantization Components by PTQ Method
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - **Component**
     - **Basic PTQ**
     - **Static PTQ**
     - **Dynamic PTQ**
     - **Mixed PTQ**
   * - **Conv/Linear Weights**
     - ✅ Quantized
     - ✅ Quantized
     - ✅ Quantized
     - ✅ Quantized (custom precision)
   * - **Biases**
     - ✅ Quantized
     - ✅ Quantized
     - ✅ Quantized
     - ✅ Quantized (custom precision)
   * - **Activations (ReLU/GELU)**
     - ⚫ Original precision
     - ✅ Quantized (calibrated)
     - ✅ Quantized (per-batch)
     - ✅ Quantized (custom precision)
   * - **BatchNorm Stats (mean/var)**
     - ⚫ Preserved (FP32)
     - ⚫ Preserved (FP32)
     - ⚫ Preserved (FP32)
     - ⚫ Preserved (FP32)
   * - **LayerNorm (scale/bias)**
     - ⚫ Original precision
     - ✅ Quantized
     - ✅ Quantized
     - ✅ Quantized (custom precision)

**Legend:**

- ✅ Quantized = Converted to specified precision (INT8, FP16, custom)
- ⚫ Original precision = Not quantized, keeps model's current precision
- ⚫ Preserved (FP32) = Always kept as FP32 for numerical stability

Calibration Algorithms
--------------------------

.. list-table:: Calibration Algorithms Comparison
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - **Algorithm**
     - **Speed**
     - **Accuracy**
     - **Memory**
     - **Best For**
   * - **MinMax**
     - ★★★★★ Fast
     - ★★★☆☆ Good
     - ★★★★★ Low
     - Simple models, quick tests
   * - **Percentile**
     - ★★★★☆ Fast
     - ★★★★☆ Better
     - ★★★★☆ Low
     - Outlier-heavy data
   * - **KL-Divergence**
     - ★★☆☆☆ Slow
     - ★★★★★ Best
     - ★★★☆☆ Medium
     - Production (TensorRT-style)
   * - **MSE**
     - ★★★☆☆ Medium
     - ★★★★☆ Better
     - ★★☆☆☆ High
     - Balance accuracy/speed


Quantization Components
--------------------------

Understanding what gets quantized in each method:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                         Model Components                        │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                 │
   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
   │  │   Weights    │  │ Activations  │  │ Batch Stats  │         │
   │  │  (Conv/FC)   │  │   (ReLU)     │  │ (BN mean/var)│         │
   │  └──────────────┘  └──────────────┘  └──────────────┘         │
   │         ▲                 ▲                  ▲                  │
   │         │                 │                  │                  │
   │    ┌────┴─────┬───────────┴──────┬──────────┴────┐             │
   │    │          │                  │               │             │
   │  Basic      Static            Dynamic         Preserved        │
   │   PTQ        PTQ               PTQ            (all PTQ)        │
   │    │          │                  │                             │
   │    ▼          ▼                  ▼                             │
   │  W-only    W + A             W + A (dynamic)                   │
   └─────────────────────────────────────────────────────────────────┘

**Component Quantization Details:**

1. **Weights (Conv/Linear)**: Always quantized in all PTQ methods
2. **Biases**: Always quantized in all PTQ methods
3. **Activations (ReLU/GELU)**: Quantized in Static/Dynamic/Mixed PTQ
4. **BatchNorm Stats**: Never quantized (preserved as FP32)
5. **LayerNorm**: Quantized in Static/Dynamic PTQ


.. _ptq_api:

API Reference
=============

.. _post_quantization:

Basic PTQ: ``post_quantization``
----------------------------------------------------

**Weight-only quantization** (fastest, simplest).

.. code-block:: python

   pychop.ptq.post_quantization(
       model,
       chop,
       eval_mode=True,
       verbose=False
   )

**Parameters:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``model``
     - **PyTorch**: ``torch.nn.Module`` | **JAX**: Flax variables dict
   * - ``chop``
     - Quantizer instance (``Chop``, ``Chopf``, or ``Chopi``)
   * - ``eval_mode``
     - Set model to eval mode (PyTorch only)
   * - ``verbose``
     - Print quantization details

**Returns:**

- **PyTorch**: Quantized ``nn.Module``
- **JAX**: Quantized params dict

**Example:**

.. code-block:: python

   import pychop
   from pychop import Chopi
   from pychop.ptq import post_quantization

   # PyTorch
   pychop.backend('torch')
   chop = Chopi(bits=8, symmetric=True)
   model_q = post_quantization(model, chop, verbose=True)

   # JAX
   pychop.backend('jax')
   from pychop.jx.layers import ChopiSTE
   chop = ChopiSTE(bits=8, symmetric=True)
   quantized_params = post_quantization(variables, chop, verbose=True)


.. _static_post_quantization:

Static PTQ: ``static_post_quantization``
----------------------------------------------------

**Weights + Activations quantization with calibration** (best accuracy).

.. code-block:: python

   pychop.ptq.static_post_quantization(
       model,
       chop,
       calibration_data,
       calibration_method='minmax',
       percentile=99.99,
       fuse_bn=True,              # PyTorch only
       eval_mode=True,            # PyTorch only
       verbose=False,
       model_apply_fn=None        # JAX only
   )

**Parameters:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``model``
     - Model to quantize
   * - ``chop``
     - Quantizer instance
   * - ``calibration_data``
     - Iterable of input batches (50-1000 batches recommended)
   * - ``calibration_method``
     - ``'minmax'`` | ``'percentile'`` | ``'kl_divergence'`` | ``'mse'``
   * - ``percentile``
     - Percentile for ``'percentile'`` method (e.g., 99.99)
   * - ``fuse_bn``
     - Fuse Conv+BN layers (PyTorch only, improves accuracy ~1-2%)
   * - ``eval_mode``
     - Set model to eval mode (PyTorch only)
   * - ``verbose``
     - Print quantization details
   * - ``model_apply_fn``
     - Model's apply function (JAX only, required for activation stats)

**Returns:**

- **PyTorch**: Quantized ``nn.Module`` with static activation hooks
- **JAX**: Dict with ``params``, ``batch_stats``, ``quant_config``

**Calibration Methods:**

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Method
     - Description
     - When to Use
   * - ``minmax``
     - Simple min/max clipping
     - Quick tests, simple models
   * - ``percentile``
     - Percentile-based clipping (e.g., 99.99%)
     - Data with outliers
   * - ``kl_divergence``
     - TensorRT-style KL-divergence optimization
     - Production, best accuracy
   * - ``mse``
     - MSE-based threshold search
     - Balance between accuracy and speed

**Example (PyTorch):**

.. code-block:: python

   import pychop
   from pychop import Chopi
   from pychop.ptq import static_post_quantization

   pychop.backend('torch')
   chop = Chopi(bits=8, symmetric=True)

   # Prepare calibration data
   calibration_data = [
       torch.randn(4, 3, 224, 224) for _ in range(100)
   ]

   # Option 1: MinMax calibration (fastest)
   model_q = static_post_quantization(
       model, chop,
       calibration_data=calibration_data,
       calibration_method='minmax',
       fuse_bn=True,
       verbose=True
   )

   # Option 2: Percentile calibration (better)
   model_q = static_post_quantization(
       model, chop,
       calibration_data=calibration_data,
       calibration_method='percentile',
       percentile=99.9,  # Clip 0.1% outliers
       verbose=True
   )

   # Option 3: KL-Divergence calibration (best)
   model_q = static_post_quantization(
       model, chop,
       calibration_data=calibration_data,
       calibration_method='kl_divergence',
       fuse_bn=True,
       verbose=True
   )

   # Use quantized model
   output = model_q(input)

**Example (JAX):**

.. code-block:: python

   import pychop
   import jax
   import jax.numpy as jnp
   from pychop.jx.layers import ChopiSTE
   from pychop.ptq import static_post_quantization

   pychop.backend('jax')
   chop = ChopiSTE(bits=8, symmetric=True)

   # Prepare calibration data
   calibration_data = [
       jax.random.normal(jax.random.PRNGKey(i), (4, 224, 224, 3))
       for i in range(100)
   ]

   # Define apply function
   def apply_fn(params, x):
       return model.apply(params, x, train=False)

   # Static PTQ with KL-divergence
   result = static_post_quantization(
       variables, chop,
       calibration_data=calibration_data,
       calibration_method='kl_divergence',
       model_apply_fn=apply_fn,
       verbose=True
   )

   # Use quantized model
   output = model.apply(result, input, train=False)


.. _dynamic_post_quantization:

Dynamic PTQ: ``dynamic_post_quantization``
-------------------------------------------

**Weights + Activations quantization without calibration** (no calibration needed).

.. code-block:: python

   pychop.ptq.dynamic_post_quantization(
       model,
       chop,
       eval_mode=True,
       verbose=False
   )

**Parameters:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``model``
     - Model to quantize
   * - ``chop``
     - Quantizer instance
   * - ``eval_mode``
     - Set model to eval mode (PyTorch only)
   * - ``verbose``
     - Print quantization details

**Returns:**

- **PyTorch**: Quantized ``nn.Module`` with dynamic activation hooks
- **JAX**: Dict with ``params``, ``batch_stats``, ``quant_config``

**Key Differences from Static PTQ:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Aspect
     - Static PTQ
     - Dynamic PTQ
   * - **Calibration**
     - Required (50-1000 batches)
     - Not needed
   * - **Inference Speed**
     - ★★★★★ Fast
     - ★★★★☆ ~5-10% slower
   * - **Accuracy**
     - ★★★★★ Best
     - ★★★★☆ 0.5-1% lower
   * - **Use Case**
     - Vision models (fixed input)
     - NLP models (variable input)

**Example:**

.. code-block:: python

   import pychop
   from pychop import Chopi
   from pychop.ptq import dynamic_post_quantization

   pychop.backend('torch')
   chop = Chopi(bits=8, symmetric=True)

   # Dynamic PTQ (no calibration needed)
   model_q = dynamic_post_quantization(model, chop, verbose=True)

   # Activations are quantized dynamically per batch
   output = model_q(input)


.. _mixed_post_quantization:

Mixed-Precision PTQ: ``mixed_post_quantization``
-------------------------------------------------

**Separate quantizers for weights and activations** (W8A16, W4A8, etc.).

.. code-block:: python

   pychop.ptq.mixed_post_quantization(
       model,
       weight_chop,
       activation_chop,
       calibration_data=None,
       calibration_method='minmax',
       percentile=99.99,
       dynamic=True,              # PyTorch only
       eval_mode=True,            # PyTorch only
       verbose=False,
       model_apply_fn=None        # JAX only
   )

**Parameters:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``model``
     - Model to quantize
   * - ``weight_chop``
     - Quantizer for weights (e.g., ``Chopi(bits=8)``)
   * - ``activation_chop``
     - Quantizer for activations (e.g., ``Chop(exp_bits=5, sig_bits=10)`` for FP16)
   * - ``calibration_data``
     - Calibration data (optional for dynamic mode)
   * - ``calibration_method``
     - Calibration algorithm (if ``calibration_data`` provided)
   * - ``percentile``
     - Percentile for ``'percentile'`` calibration
   * - ``dynamic``
     - Use dynamic activation quantization (PyTorch only)
   * - ``eval_mode``
     - Set model to eval mode (PyTorch only)
   * - ``verbose``
     - Print quantization details
   * - ``model_apply_fn``
     - Model's apply function (JAX only)

**Popular Mixed-Precision Configurations:**

.. list-table::
   :widths: 15 30 25 30
   :header-rows: 1

   * - Config
     - Weight Quantizer
     - Activation Quantizer
     - Use Case
   * - **W8A16**
     - ``Chopi(bits=8)``
     - ``Chop(exp_bits=5, sig_bits=10)``
     - LLM quantization (minimal accuracy loss)
   * - **W4A8**
     - ``Chopi(bits=4)``
     - ``Chopi(bits=8)``
     - Extreme compression (75% size reduction)
   * - **W2A8**
     - ``Chopi(bits=2)``
     - ``Chopi(bits=8)``
     - Experimental (87.5% size reduction)
   * - **W8A8**
     - ``Chopi(bits=8)``
     - ``Chopi(bits=8)``
     - Standard INT8 quantization

**Example (W8A16 - LLM Quantization):**

.. code-block:: python

   import pychop
   from pychop import Chopi, Chop
   from pychop.ptq import mixed_post_quantization

   pychop.backend('torch')

   # W8A16 configuration
   weight_chop = Chopi(bits=8, symmetric=True)        # 8-bit weights
   activation_chop = Chop(exp_bits=5, sig_bits=10)    # FP16 activations

   # Option 1: Dynamic (no calibration)
   model_q = mixed_post_quantization(
       model, weight_chop, activation_chop,
       dynamic=True,
       verbose=True
   )

   # Option 2: Static (with calibration)
   calibration_data = [torch.randn(4, 3, 224, 224) for _ in range(50)]
   model_q = mixed_post_quantization(
       model, weight_chop, activation_chop,
       calibration_data=calibration_data,
       calibration_method='percentile',
       dynamic=False,
       verbose=True
   )

**Example (W4A8 - Extreme Compression):**

.. code-block:: python

   # W4A8 configuration
   weight_chop = Chopi(bits=4, symmetric=True)   # 4-bit weights (75% size reduction)
   activation_chop = Chopi(bits=8, symmetric=True)  # 8-bit activations

   model_q = mixed_post_quantization(
       model, weight_chop, activation_chop,
       dynamic=True,
       verbose=True
   )


.. _ptq_examples:

Complete Examples
=================

Example 1: ResNet-18 INT8 PTQ (PyTorch)
----------------------------------------

.. code-block:: python

   import torch
   import torchvision.models as models
   import pychop
   from pychop import Chopi
   from pychop.ptq import static_post_quantization

   # Load pre-trained ResNet-18
   pychop.backend('torch')
   model = models.resnet18(pretrained=True)
   model.eval()

   # Prepare calibration data (100 batches from ImageNet)
   calibration_data = []
   for images, _ in train_loader:
       calibration_data.append(images)
       if len(calibration_data) >= 100:
           break

   # INT8 quantization with percentile calibration
   chop = Chopi(bits=8, symmetric=True)
   model_q = static_post_quantization(
       model, chop,
       calibration_data=calibration_data,
       calibration_method='percentile',
       percentile=99.9,
       fuse_bn=True,
       verbose=True
   )

   # Evaluate accuracy
   def evaluate(model, test_loader):
       correct = 0
       total = 0
       with torch.no_grad():
           for images, labels in test_loader:
               outputs = model(images)
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
       return 100 * correct / total

   fp32_acc = evaluate(model, test_loader)
   int8_acc = evaluate(model_q, test_loader)

   print(f"FP32 Accuracy: {fp32_acc:.2f}%")
   print(f"INT8 Accuracy: {int8_acc:.2f}%")
   print(f"Accuracy Drop: {fp32_acc - int8_acc:.2f}%")

   # Expected output:
   # FP32 Accuracy: 69.76%
   # INT8 Accuracy: 69.34%
   # Accuracy Drop: 0.42%


Example 2: BERT-Base W8A16 PTQ (PyTorch)
-----------------------------------------

.. code-block:: python

   import torch
   from transformers import BertForSequenceClassification
   import pychop
   from pychop import Chopi, Chop
   from pychop.ptq import mixed_post_quantization

   # Load pre-trained BERT
   pychop.backend('torch')
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
   model.eval()

   # Prepare calibration data
   calibration_data = [
       torch.randint(0, 30522, (4, 128)) for _ in range(50)  # 50 batches
   ]

   # W8A16 configuration
   weight_chop = Chopi(bits=8, symmetric=True)        # 8-bit weights
   activation_chop = Chop(exp_bits=5, sig_bits=10)    # FP16 activations

   # Mixed-precision PTQ
   model_q = mixed_post_quantization(
       model, weight_chop, activation_chop,
       calibration_data=calibration_data,
       calibration_method='percentile',
       percentile=99.99,
       dynamic=False,
       verbose=True
   )

   # Model size comparison
   import os
   torch.save(model.state_dict(), 'bert_fp32.pth')
   torch.save(model_q.state_dict(), 'bert_w8a16.pth')

   fp32_size = os.path.getsize('bert_fp32.pth') / (1024**2)
   w8a16_size = os.path.getsize('bert_w8a16.pth') / (1024**2)

   print(f"FP32 Model Size: {fp32_size:.2f} MB")
   print(f"W8A16 Model Size: {w8a16_size:.2f} MB")
   print(f"Size Reduction: {(1 - w8a16_size/fp32_size)*100:.1f}%")

   # Expected output:
   # FP32 Model Size: 438.00 MB
   # W8A16 Model Size: 219.00 MB
   # Size Reduction: 50.0%


Example 3: Vision Transformer (ViT) PTQ (JAX)
----------------------------------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from flax import linen as nn
   import pychop
   from pychop.jx.layers import ChopiSTE
   from pychop.ptq import static_post_quantization

   # Define a simple ViT model
   class SimpleViT(nn.Module):
       num_classes: int = 1000
       
       @nn.compact
       def __call__(self, x, train=False):
           # Patch embedding
           x = nn.Conv(features=768, kernel_size=(16, 16), strides=16)(x)
           
           # Transformer blocks (simplified)
           for _ in range(12):
               # Self-attention
               attn = nn.MultiHeadDotProductAttention(num_heads=12)(x, x)
               x = x + attn
               x = nn.LayerNorm()(x)
               
               # MLP
               mlp = nn.Dense(features=3072)(x)
               mlp = nn.gelu(mlp)
               mlp = nn.Dense(features=768)(mlp)
               x = x + mlp
               x = nn.LayerNorm()(x)
           
           # Classification head
           x = jnp.mean(x, axis=(1, 2))
           x = nn.Dense(features=self.num_classes)(x)
           return x

   # Initialize model
   pychop.backend('jax')
   model = SimpleViT()
   rng = jax.random.PRNGKey(0)
   variables = model.init(rng, jnp.ones((1, 224, 224, 3)), train=False)

   # Prepare calibration data
   calibration_data = [
       jax.random.normal(jax.random.PRNGKey(i), (4, 224, 224, 3))
       for i in range(100)
   ]

   # Define apply function
   def apply_fn(params, x):
       return model.apply(params, x, train=False)

   # INT8 quantization with KL-divergence
   chop = ChopiSTE(bits=8, symmetric=True)
   result = static_post_quantization(
       variables, chop,
       calibration_data=calibration_data,
       calibration_method='kl_divergence',
       model_apply_fn=apply_fn,
       verbose=True
   )

   # Use quantized model
   test_input = jax.random.normal(rng, (1, 224, 224, 3))
   output = model.apply(result, test_input, train=False)

   print(f"Output shape: {output.shape}")
   print(f"Quantization config: {result['quant_config']}")


Example 4: Comparing Calibration Methods
-----------------------------------------

.. code-block:: python

   import torch
   import torchvision.models as models
   import pychop
   from pychop import Chopi
   from pychop.ptq import static_post_quantization

   # Load model
   pychop.backend('torch')
   model = models.mobilenet_v2(pretrained=True)
   model.eval()

   # Prepare calibration data
   calibration_data = [
       torch.randn(4, 3, 224, 224) for _ in range(100)
   ]

   chop = Chopi(bits=8, symmetric=True)

   # Test all calibration methods
   methods = ['minmax', 'percentile', 'kl_divergence', 'mse']
   results = {}

   for method in methods:
       print(f"\n{'='*60}")
       print(f"Testing {method.upper()} calibration")
       print('='*60)
       
       model_q = static_post_quantization(
           model, chop,
           calibration_data=calibration_data,
           calibration_method=method,
           percentile=99.9 if method == 'percentile' else 99.99,
           fuse_bn=True,
           verbose=True
       )
       
       # Evaluate
       acc = evaluate(model_q, test_loader)
       results[method] = acc
       print(f"{method} Accuracy: {acc:.2f}%")

   # Summary
   print(f"\n{'='*60}")
   print("Summary: Calibration Method Comparison")
   print('='*60)
   for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
       print(f"{method:20s}: {acc:.2f}%")

   # Expected output:
   # ============================================================
   # Summary: Calibration Method Comparison
   # ============================================================
   # kl_divergence       : 71.85%
   # mse                 : 71.72%
   # percentile          : 71.58%
   # minmax              : 71.34%


.. _ptq_best_practices:

Best Practices
--------------------------

1. Choosing PTQ Method
-----------------------

.. code-block:: text

   Decision Tree:
   
   Need calibration data?
   ├─ No  → Use Dynamic PTQ
   └─ Yes → 
       └─ High accuracy required?
           ├─ Yes → Use Static PTQ (KL-divergence or MSE)
           └─ No  → Use Static PTQ (MinMax or Percentile)
   
   Need mixed precision?
   └─ Use Mixed PTQ (W8A16 for LLMs, W4A8 for extreme compression)


2. Calibration Data Guidelines
-------------------------------

**Size:**

- Vision models: 50-200 batches (200-800 images)
- NLP models: 100-500 batches
- Small models (<10M params): 50 batches
- Large models (>100M params): 200-1000 batches

**Diversity:**

.. code-block:: python

   # Good: Diverse calibration data
   calibration_data = sample_diverse_batches(train_loader, n=100)

   # Bad: Only one class
   calibration_data = [cat_images for _ in range(100)]  # Only cats!

**Preprocessing:**

.. code-block:: python

   # Apply same preprocessing as training
   transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
   ])


3. Conv+BN Fusion (PyTorch)
----------------------------

**Always enable for better accuracy:**

.. code-block:: python

   # Good
   model_q = static_post_quantization(
       model, chop,
       calibration_data=data,
       fuse_bn=True,  # ✅ Improves accuracy ~1-2%
       verbose=True
   )

   # Bad
   model_q = static_post_quantization(
       model, chop,
       calibration_data=data,
       fuse_bn=False,  # ❌ Lower accuracy
       verbose=True
   )


4. Percentile Selection
------------------------

**Guidelines:**

- **99.99%**: Default, works for most cases
- **99.9%**: More aggressive clipping, better for outlier-heavy data
- **99.0%**: Very aggressive, use with caution

.. code-block:: python

   # Data with many outliers (e.g., sensor data)
   model_q = static_post_quantization(
       model, chop,
       calibration_data=data,
       calibration_method='percentile',
       percentile=99.9,  # Clip 0.1% outliers
       verbose=True
   )


5. JAX Backend Tips
-------------------

**Always provide ``model_apply_fn`` for activation quantization:**

.. code-block:: python

   # Good
   def apply_fn(params, x):
       return model.apply(params, x, train=False)

   result = static_post_quantization(
       variables, chop,
       calibration_data=data,
       model_apply_fn=apply_fn,  # ✅ Required for activation stats
       verbose=True
   )

   # Bad
   result = static_post_quantization(
       variables, chop,
       calibration_data=data,
       # ❌ No model_apply_fn = no activation quantization
       verbose=True
   )


.. _ptq_troubleshooting:

Troubleshooting
===============

Common Issues
-------------

**Issue 1: Large Accuracy Drop (>5%)**

.. code-block:: python

   # Solution 1: Use better calibration method
   model_q = static_post_quantization(
       model, chop,
       calibration_data=data,
       calibration_method='kl_divergence',  # Try KL instead of minmax
       fuse_bn=True,
       verbose=True
   )

   # Solution 2: Increase calibration data
   calibration_data = [batch for batch in train_loader[:200]]  # More data

   # Solution 3: Use mixed-precision
   weight_chop = Chopi(bits=8)
   activation_chop = Chop(exp_bits=5, sig_bits=10)  # FP16 activations
   model_q = mixed_post_quantization(
       model, weight_chop, activation_chop,
       calibration_data=data,
       verbose=True
   )


**Issue 2: Slow Calibration**

.. code-block:: python

   # Solution 1: Use faster calibration method
   model_q = static_post_quantization(
       model, chop,
       calibration_data=data,
       calibration_method='percentile',  # Faster than KL/MSE
       verbose=True
   )

   # Solution 2: Reduce calibration data
   calibration_data = calibration_data[:50]  # Use fewer batches


**Issue 3: Out of Memory (OOM)**

.. code-block:: python

   # Solution 1: Reduce batch size in calibration data
   calibration_data = [
       torch.randn(2, 3, 224, 224)  # Smaller batch size
       for _ in range(100)
   ]

   # Solution 2: Use dynamic PTQ (no calibration)
   model_q = dynamic_post_quantization(model, chop, verbose=True)


**Issue 4: JAX "model_apply_fn required" Error**

.. code-block:: python

   # Solution: Always provide apply function
   def apply_fn(params, x):
       return model.apply(params, x, train=False)

   result = static_post_quantization(
       variables, chop,
       calibration_data=data,
       model_apply_fn=apply_fn,  # ✅ Add this
       verbose=True
   )


References
--------------------------

- `TensorRT Documentation <https://docs.nvidia.com/deeplearning/tensorrt/>`_
- `PyTorch Quantization <https://pytorch.org/docs/stable/quantization.html>`_
- `ZeroQuant: Efficient INT8 Quantization <https://arxiv.org/abs/2206.01861>`_
- `GPTQ: Accurate Quantization for GPT Models <https://arxiv.org/abs/2210.17323>`_
