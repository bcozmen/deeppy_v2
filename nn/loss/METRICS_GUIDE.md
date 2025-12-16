# Contrastive Loss Metrics Guide

This document describes the metrics available in `SupConLoss` and `HMLC` classes for evaluating similarity matrices during training.

## Usage

Enable metrics by setting `return_metrics=True`:

```python
# For SupConLoss
criterion = SupConLoss(temperature=0.07, return_metrics=True)
loss, metrics = criterion(features, labels=labels)

# For HMLC
criterion = HMLC(temperature=0.05, return_metrics=True)
loss, layer_losses, layer_metrics = criterion(features, labels)
```

## Available Metrics

### 1. Positive Pair Statistics

Measures similarities between samples that **should** be close (same class).

- **`pos_sim_mean`**: Average cosine similarity of positive pairs
  - Range: [-1, 1], typically [0, 1] after training
  - **Target**: High (close to 1)
  
- **`pos_sim_std`**: Standard deviation of positive similarities
  - Indicates consistency within classes
  - Lower std = more compact clusters

- **`pos_sim_min` / `pos_sim_max`**: Range of positive similarities
  - Helps identify outliers or difficult samples

- **`num_positives`**: Total number of positive pairs evaluated

### 2. Negative Pair Statistics

Measures similarities between samples that **should** be far apart (different classes).

- **`neg_sim_mean`**: Average cosine similarity of negative pairs
  - Range: [-1, 1], typically [-0.5, 0.5]
  - **Target**: Low (close to 0 or negative)

- **`neg_sim_std`**: Standard deviation of negative similarities
  - High std might indicate some classes are harder to separate

- **`neg_sim_min` / `neg_sim_max`**: Range of negative similarities
  - `neg_sim_max` close to 1 indicates confusion between some classes

- **`num_negatives`**: Total number of negative pairs evaluated

### 3. Separation Metrics

#### **`pos_neg_gap`**
Simple difference: `pos_sim_mean - neg_sim_mean`

- **Range**: [-2, 2], typically [0, 1] during training
- **Target**: Large positive gap (> 0.5 is good)
- **Interpretation**: Larger gap = better class separation

#### **`alignment`** (Wang & Isola, 2020)
Measures how well positive pairs align in feature space.

$$
\text{alignment} = -\mathbb{E}_{(x, x^+)} [(x - x^+)^2] = -\mathbb{E}[2 - 2\cdot \text{sim}(x, x^+)]
$$

- **Range**: Approximately [-4, 0]
- **Target**: Higher is better (closer to 0)
- **Interpretation**: 
  - Close to 0: Perfect alignment, positive pairs are identical
  - Close to -4: Poor alignment, positive pairs are orthogonal

#### **`uniformity`** (Wang & Isola, 2020)
Measures how uniformly features are distributed on the hypersphere.

$$
\text{uniformity} = \log \mathbb{E}_{x, y} [e^{-2\|x - y\|^2}]
$$

- **Range**: Typically [-5, 0] 
- **Target**: Lower is better (< -2 is good)
- **Interpretation**:
  - Low (very negative): Features spread uniformly
  - High (close to 0): Features collapse to few points (dimensional collapse)
  - Prevents representation collapse

#### **`pos_rank_mean`**
Average rank of positive pairs when sorting all pairs by similarity.

- **Range**: [1, N] where N = number of samples
- **Target**: Low (ideally close to 1)
- **Interpretation**:
  - Rank 1: Positive is the most similar sample
  - Rank N: Positive is the least similar sample
  - Measures retrieval quality

## Interpretation Guidelines

### Healthy Training Indicators

1. **`pos_sim_mean` > 0.7**: Strong intra-class cohesion
2. **`neg_sim_mean` < 0.3**: Good inter-class separation  
3. **`pos_neg_gap` > 0.5**: Clear decision boundary
4. **`alignment` > -1.0**: Positive pairs are well-aligned
5. **`uniformity` < -2.0**: No dimensional collapse
6. **`pos_rank_mean` < 5**: Positives consistently rank highly

### Warning Signs

⚠️ **High `neg_sim_max` (> 0.8)**: Some classes are very similar, might need:
   - More training
   - Better data augmentation
   - Architecture changes

⚠️ **Low `pos_sim_min` (< 0.3)**: Some samples are outliers, might be:
   - Mislabeled data
   - Hard samples
   - Need for outlier detection

⚠️ **High `uniformity` (> -1.0)**: Dimensional collapse occurring:
   - Features using only a few dimensions
   - Reduce temperature or adjust architecture

⚠️ **High `pos_rank_mean` (> 10)**: Poor retrieval performance:
   - Model struggles to distinguish positives from negatives
   - May need more training or different loss

## HMLC-Specific Considerations

For hierarchical labels, metrics are computed **per layer**:

- **Early layers** (coarse labels): Expect higher pos_sim_mean, simpler separation
- **Later layers** (fine labels): Expect lower pos_sim_mean, harder separation
- **Progression**: Metrics should gradually worsen from coarse to fine layers

Example progression for taxonomy (Animal → Mammal → Dog):
```
Layer 1 (Animal level):
  pos_neg_gap: 0.8  (easy: animals vs non-animals)
  
Layer 2 (Mammal level):  
  pos_neg_gap: 0.5  (medium: mammals vs birds within animals)
  
Layer 3 (Species level):
  pos_neg_gap: 0.3  (hard: dogs vs cats within mammals)
```

## References

- **SupCon Paper**: [Supervised Contrastive Learning (Khosla et al., 2020)](https://arxiv.org/abs/2004.11362)
- **Alignment & Uniformity**: [Understanding Contrastive Representation Learning (Wang & Isola, 2020)](https://arxiv.org/abs/2005.10242)

## Example Monitoring During Training

```python
# Log metrics every epoch
loss, metrics = criterion(features, labels)

logger.log({
    'loss': loss.item(),
    'pos_sim': metrics['pos_sim_mean'],
    'neg_sim': metrics['neg_sim_mean'],
    'gap': metrics['pos_neg_gap'],
    'alignment': metrics['alignment'],
    'uniformity': metrics['uniformity'],
})

# Early stopping based on metrics
if metrics['pos_neg_gap'] > 0.7 and metrics['uniformity'] < -2.5:
    print("Excellent separation achieved!")
```

## Visualization Suggestions

1. **Similarity Heatmap**: Visualize the full similarity matrix
2. **Distribution Plots**: Histogram of pos_sim vs neg_sim
3. **Metric Curves**: Track alignment, uniformity over training
4. **Per-class Analysis**: Compute metrics separately for each class
5. **t-SNE/UMAP**: Project features to 2D to verify clustering

## Computing Additional Custom Metrics

You can compute additional metrics by accessing intermediate values:

```python
# Inside your training loop
loss, metrics = criterion(features, labels)

# Add custom metrics
metrics['intra_class_variance'] = compute_intra_class_var(features, labels)
metrics['silhouette_score'] = compute_silhouette(features, labels)
```
