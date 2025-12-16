# AMP NaN Fix Summary

## Problem
NaN values appearing after ~30k training steps when using Automatic Mixed Precision (AMP).

## Root Causes Identified

### 1. **FP16 Underflow in Gate Normalization** (`gmn.py`)
- Using epsilon values like `1e-8` and `1e-10` causes underflow in FP16
- Division by very small normalized factors amplifies errors

### 2. **FP16 Overflow in Exponential Operations** (`gmn.py`)
- `torch.exp()` on gates can overflow in FP16 without proper stabilization

### 3. **Unsafe Logarithm Operations** (`gmn.py`, `pooling.py`, `contrastive.py`)
- `torch.log()` on small FP16 values produces -inf
- Adding small epsilon (1e-10, 1e-12) doesn't prevent underflow in FP16

### 4. **Division by Small Sums** (`contrastive.py`)
- Mask sums can be very small in FP16, causing NaN when used as denominators

## Fixes Applied

### 1. **gmn.py - `_transform_gates()`**
- Promote gates to FP32 before exp/softmax operations
- Increase epsilon from `1e-8` to `1e-6` (safe for FP32)
- Convert result back to original dtype

### 2. **gmn.py - `_calculate_normalized_entropy()`**
- Keep entropy computation in FP32 
- Use `torch.clamp(normalized_gates, min=1e-6)` instead of adding small epsilon
- Use `torch.clamp(torch.log(...), min=1e-6)` for safe log operations
- Removed unnecessary dtype conversions

### 3. **gmn.py - `_normalize_gates()`**
- Promote to FP32 for normalization
- Increase epsilon from `1e-8` to `1e-6`
- Convert back to original dtype

### 4. **pooling.py - `AttentionPooling.forward()`**
- Always compute attention in FP32
- Use `torch.clamp(attn, min=1e-6)` before log operation
- Simplified by removing conditional FP16 checks
- Pool in FP32, convert result back to original dtype

### 5. **pooling.py - `MultiQueryAttentionPooling.forward()`**
- Always compute in FP32 (simpler and more stable)
- Use `torch.clamp(attn, min=1e-6)` for entropy
- Removed complex conditional FP16 logic

### 6. **contrastive.py - `SupConLoss.forward()`**
- Promote critical operations to FP32
- Add epsilon `1e-6` to log denominator
- Use `torch.clamp(mask.sum(1), min=1e-6)` to prevent division by zero

## Key Principles

1. **Always use FP32 for:**
   - Softmax operations
   - Logarithm operations
   - Division by potentially small values
   - Entropy calculations

2. **Use clamping instead of small epsilon:**
   - `torch.clamp(x, min=1e-6)` before `log(x)`
   - Prevents both underflow and -inf

3. **Epsilon values:**
   - FP32: Use `1e-6` (safe range)
   - FP16: Too small epsilons (< 1e-5) cause underflow

4. **Minimal overhead:**
   - Only promote to FP32 where numerically critical
   - Convert back to original dtype for memory efficiency

## Expected Results

- **No more NaN values** during extended training (30k+ steps)
- **Stable gradient flow** through attention and entropy computations
- **Minimal performance impact** (only critical ops promoted to FP32)
- **Maintained accuracy** while gaining AMP speedup benefits
