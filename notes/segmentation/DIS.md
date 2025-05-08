# [DIS](https://arxiv.org/pdf/2203.03041)

Dichotomous Image Segmentation (DIS) proposed **IS-Net**. IS-Net as 3 components

1. ground truth (GT) encoder,
2. image segmentation component ($U^2$-Net with an input convolution layer before its first encoder stage.)
3. intermediate supervision strategy

## 1st Stage

Self-supervised training of the GT-encoder

$L_{gt} = \sum_{d=1}^{D} \lambda_{d}^{gt} BCE(F_{gt}(\theta_{gt}, G)_d, G)$

GT encoder will be frozen.

## 2nd Stage
High dimensional intermediate features

$f_{D}^{G} = F_{gt}^{-}(\theta_{gt}, G), D= \{1,2,3,4,5,6\}$ 

- $F_{gt}^{-}$ represents the $F_{gt}$  without the last convolution layers for generating the probability maps.

- $F_{gt}^{-}$ is to supervise those corresponding features $f_{D}^{I}$ from the segmentation model $F_{sg}$


High dimensional intermediate features from image segmentation component

$f_{I}^{G} = F_{sg}^{-}(\theta_{sg}, G), D= \{1,2,3,4,5,6\}$ 


Feature Consistency Loss (intermediate supervision)

$L_{fs} = \sum_{d=1}^{D} \lambda_{d}^{fs} ||f_{d}^{I} - f_{d}^{G}||^2$

$L_{sg} = \sum_{d=1}^{D} \lambda_{d}^{sg} BCE(F_{sg}(\theta_{sg}, I), G)$

Loss for $F_{sg}$ is  $L = L_{fs} + L_{sg}$

<div align='center'>
<img src='../../assets/ISNet.png' height=400 width=600>
<figcaption>
IS-Net
</figcaption>
</div>

## Results

<div align='center'>
<img src='../../assets/DIS_Result.png' height=200 width=600>
<figcaption>
Result
</figcaption>
</div>

## Meric : Human Correction Efforts (HCE)

<div align='center'>
<img src='../../assets/Relax_HCE_Algo.png' height=300 width=400>
<figcaption>
</figcaption>
</div>