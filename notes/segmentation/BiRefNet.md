# [BiRefNet](https://arxiv.org/pdf/2401.03407)
Bilateral Reference for High-Resolution Dichotomous Image Segmentation

- Swin-Transformer Based
- inward reference(InRef) and an outward reference (OutRef)

Two essential Module:

1. Localization module (LM) : 
    - object localization using global semantic information
    - extract hierarchical features from vision transformer backbone, which are combined and squeezed to obtain corase predictions in low resolution in deep layers.
2. Reconstruction module (RM) : 
    - hierarchical patches of images provide the source reference, and gradient maps serve as the target reference.
    - the inward and outward references as bilateral references (BiRef), in which the source image and the gradient map are fed into the decoder at different stages.

<div align='center'>
<img src='../../assets/BiRefNetComparison.png' height=200 width=600>
<figcaption>
BiRefNet Comparison
</figcaption>
</div>


## Localization Module
<div align='center'>
<img src='../../assets/BiRefNet.png' height=300 width=600>
<figcaption>
BiRefNet Architecture
</figcaption>
</div>


- Transformer Encoder extract the fearures at different stages i.e., $F_1^e, F_2^e, F_3^e$ with resolution at 4,8,16,32. 
- The features of the first four $\{F_i^e\}_{i=1}^3$ are transferred to the corresponding decoder stages with lateral connections (1×1 convolution layers).
- These features are stacked and concatenated in the last encoder block to generate $F^e$ then fed into a classification module.

To enlarge the receptive fields to cover features of large objects and focus on local features for high precision simultaneously [Atrous Spatial Pyramid Pooling (ASPP)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf)  is used for multi-context fusion.


## Reconstruction Module
<div align='center'>
<img src='../../assets/BiRefNetBlock.png' height=150 width=600>
<figcaption>
BiRef Blocks
</figcaption>
</div>

- Small receptive field (RFs) lead to inadequate context information to locate the right target on a large background, whereas large RFs often result in insufficient feature extraction in detailed areas.

- To achieve balance, using reconstruction block (RB) in each **BiRef** block as a replacement for the vanilla residual blocks.

- In RB, we employ [deformable convolutions](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf) with hierarchical receptive fields (i.e., 1×1, 3×3, 7×7) and an adaptive average pooling layer to extract features with RFs of various scales.

- These features extracted by different RFs are then concatenated as $F_i^{\theta}$, followed by a 1×1 convolution layer and a batch normalization layer to generate the output feature of RM $F_i^{d'}$.


## Bilateral Reference
- inward reference(InRef) and an outward reference (OutRef)

- In InRef, images $I$ with original high resolution are cropped to patches $\{P_{k=1}^N\}$ of consistent size with the output features of the corresponding decoder stage.

- These patches are stacked with the original feature $F_i^{d+}$ to be fed into the RM.

- In OutRef, we use gradient labels to draw more attention to areas of richer gradient information which is essential for the segmentation of fine structures.

- First, we extract the gradient maps of the input images as $G_i^{gt}$. Meanwhile, $F_i^{\theta}$
is used to generate the feature $F_i^G$ to produce the predicted
gradient maps $\hat{G}^i$

- It passes through a conv and a sigmoid layer and is used to generate the gradient referring attention $A_i^G$, which is then multiplied by $F_i^{d'}$ to generate output of the BiRef block as $F_{i−1}^{d}$.

## Loss

$L = L_{pixel} + L_{region} + L_{boundary} + L_{semantic} \\
= \lambda_1 L_{BCE} + \lambda_2 L_{IoU} + \lambda_3 L_{SSIM} + \lambda_4 L_{CE}$

$L_{BCE} = -\sum_{(i,j)} [G(i,j) \log(M(i,j)) + (1-G(i,j)) \log(1-M(i,j))]$

$L_{IoU} = 1 - \frac{\sum_{r=1}^H \sum_{c=1}^W M(i,j)G(i,j)}{\sum_{r=1}^H \sum_{c=1}^W [M(i,j)+G(i,j)-M(i,j)G(i,j)]}$

$L_{SSIM} = 1 - \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$

$L_{CE} = -\sum_{c=1}^N y_{o,c}\log(p_{o,c})$