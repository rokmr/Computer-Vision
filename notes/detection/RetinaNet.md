# [RetinaNet](https://arxiv.org/pdf/1708.02002)

Since there are lots of anchor box in which there is no object and very few of them object. We need to incorporate this information in the loss function which is done with the weighting of the loss function.


<div align='center'>
<img src='../../assets/FocalLoss.png' height=200>
<figcaption>
Focal Loss
</figcaption>
</div>

As $\gamma$ increases the easy sample weight decreases.

## Key Point
- Proposed: Focal loss
- Powerful feature extraction: ResNet
- Multi-scale prediction
- 9 anchors per level, each one with a classification and
regression target