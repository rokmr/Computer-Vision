# [DETR](https://arxiv.org/pdf/2005.12872)
<div>
<img src='../../assets/DETRIntro.png'>
</div>

A direct set prediction approach to bypass the surrogate tasks (like proposals, anchors, window centers, non-maximum suppression).

A encoder-decoder based architecure.

It predicts all objects at once, and is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects.

## Components

### Set Prediction

`Bipartite matching loss`

The usual solution is to design a loss based on the Hungarian algorithm, to find a bipartite matching between ground-truth and prediction. This enforces permutation-invariance, and guarantees that each target element has a unique match.

### Transformers and Parallel Decoding
Combine transformers and parallel decoding for their suitable trade-off between computational cost and the ability to perform the global computations required for set prediction.

### Object detection
1. Two-Stage detectors
2. One-Stage detectors

The final performance of above systems heavily depends on the exact way these initial guesses are set.

## Model
<div>
<img src='../../assets/DETRArch.png'>
</div>

### Object detection set prediction loss
DETR infers a fixed-size set of N predictions, in a single pass through the decoder, where N is set to be significantly larger than the typical number of objects in an image.

$y:$ ground-truth ($y_i = (c_i, b_i)$, $c_i$ rget class label and $b_i \in [0,1]^4$ is ground-truth box.)

$\hat{y} = \{\hat{y}_i\}_{i=1}^{N}:$ set of N predictions

$y$ also as a set of size $N$ padded with $\varnothing$ (no object)

Permutation of element with lowest cost:

$\hat{\sigma} = \arg\min_{\sigma\in\mathfrak{G}_N} \sum_{i}^N \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$

**Pair-wise matching cost**

$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i\neq\varnothing\}}\hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i\neq\varnothing\}}\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$

$\hat{p}_{\sigma_i}(c_i):$ prediction with index $\sigma(i)$ we define probability of class $c_i$

**Hungarian loss**

$\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^N \left[-\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i\neq\varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)})\right]$

**Box-Loss**

$\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)}) = \lambda_{\text{iou}}\mathcal{L}_{\text{iou}}(b_i, \hat{b}_{\sigma(i)}) + \lambda_{\text{L1}}\|b_i - \hat{b}_{\sigma(i)}\|_1$


### DETR architecture

**CNN Backbone**

As a feature extractor

**Transformer encoder**

Since the transformer architecture is permutation-invariant, we supplement it with fixed positional encodings that are added to the input of each attention layer.

**Transformer decoder**

The difference with the original transformer is that our model decodes the N objects in parallel at each decoder layer, while Vaswani et al use an autoregressive model that predicts the output sequence one element at a time.

**Prediction feed-forward networks (FFNs)**

The final prediction is computed by a 3-layer perceptron with ReLU activation function and hidden dimension d, and a linear projection layer.

<div>
<img src='../../assets/DETRResults.png'>
</div>

Visualization of all box predictions on all images from COCO 2017 val set for 20 out of total N = 100 prediction slots in DETR decoder. 

This shows that the prediction box focuses on different part of the image.


# RT-DETR

# Fast-DETR

# DETRs-FastDETR

# SAM-Det

# ULTRA-DETR