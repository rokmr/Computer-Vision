# [YOLO-World](https://arxiv.org/pdf/2401.17270)

- YOLO with openvocabulary detection capabilities through vision-language
modeling and pre-training on large-scale datasets.

- Propose a new Re-parameterizable VisionLanguage Path Aggregation Network (RepVL-PAN) and region-text contrastive loss to facilitate the interaction between visual and linguistic information.

- $\Omega = \{B_i, t_i \}_{i=1}^{N}$ where $t_i$ is the corresponding text for the region $B_i$

<div align='center'>
<img src='../../assets/Comparison_with_Detection_Paradigm.png' height=200>
<figcaption>
YOLO-World Architecture
</figcaption>
</div>

<div align='center'>
<img src='../../assets/YOLO-World_Arch.png' height=250>
<figcaption>
YOLO-World Architecture
</figcaption>
</div>

**Architecture**
1. YOLO Detector 
    - YOLOv8
2. Text Encoder 
    - CLIP
3. Text Contrastive Head 
    - object-text similarity
    - $s_{k,j} = \alpha \cdot \text{L2-Norm}(e_k) \cdot \text{L2-Norm}(w_j)^T + \beta,$
    - $e_k$ object embedding and $w_j$ word embedding
    - $\alpha$ and $\beta$ are learnable.

## Re-parameterizable Vision-Language PAN
- establish the feature pyramids {P3,P4,P5} with the
multi-scale image features {C3,C4,C5}
- propose the Text-guided CSPLayer (T-CSPLayer) and Image-Pooling Attention (I-Pooling Attention) to further enhance the interaction between image features and text features, which can improve the visual-semantic representation for open-vocabulary capability

<div align='center'>
<img src='../../assets/RepVL-PAN.png' height=250>
<figcaption>
RepVL-PAN
</figcaption>
</div>

### Text-guided CSPLayer

Text-guided cross-stage partial layers

- text embeddings W
- Image features $X_l \in R^{H \times W \times D} (l \in \{3, 4, 5 \})$
- $\delta$ sigmoid function
$X_l' = X_l \cdot \delta(\max_{j \in \{1..C\}} (X_lW_j^{\top}))$

### Image-Pooling Attention

- Max-Pooling output $\tilde{X}$

$W' = W + \text{MultiHead-Attention}(W, \tilde{X}, \tilde{X})$

## Training

$\mathcal{L}(I) = \mathcal{L}_{con} + \lambda_I \cdot (\mathcal{L}_{iou} + \mathcal{L}_{dfl})$

- $\mathcal{L}_{con}$ : region-text contrastive loss
- $\mathcal{L}_{dfl}$ : distributed focal loss
- $\lambda_I \in \{0, 1\}$ 
    - 1 if $I$ is from detection or grounding data
    - 0 if $I$ is from the image-text data