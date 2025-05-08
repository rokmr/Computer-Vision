# Face Detection

## Haar Cascade
Detail are present [here](ObjectDetection.md)

## HoG + SVM
Detail are present [here](ObjectDetection.md)

## [FaceNet](https://arxiv.org/pdf/1503.03832)

<div align='center'>
<img src='../assets/FaceNet.png' height=150>
<figcaption>
FaceNet
</figcaption>
</div>


<div align='center'>
<img src='../assets/TripletLoss.png' height=150>
<figcaption>
Triplet Loss
</figcaption>
</div>

### Loss
$L = \sum_{i}^{N} \left[\|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha\right]_+$
