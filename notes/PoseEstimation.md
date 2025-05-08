# Pose Estimation
- Estimate a 2D pose (x,y) coordinates for each joint from a RGB image.
- 17 joints
- Challanges
    - occlusions
    - clothing
    - extreme poses
    - viewpoint changes etc.

## [Direct Regression](https://arxiv.org/pdf/1312.4659)

<div align='center'>
<img src='../assets/DeepPose.png' height=200>
<figcaption>
DeepPose
</figcaption>
</div>



## [HeatMap Predicion](https://arxiv.org/pdf/1603.06937)
- Instead of prediction by regression, for each joint one
predicts a full image with a heatmap of the joint location
- Powerful representation, easier to predict a confidence per
location, rather than regress a value for the position
- Ground truth (GT) heatmap is constructed by placing a 2D
Gaussian around the joint position (e.g. variance 1.5 pixels)
- Loss: MSE between predicted and GT heatmap

<div align='center'>
<img src='../assets/Newell.png' height=150>
<figcaption>
Newell predicted heatmap
</figcaption>
</div>

Bringing the structure of the problem
- Body parts are linked to each other
- Body symmetries
- Joint limits, e.g., elbow cannot bend backwards
- Physical connectivity: elbow connected to wrist

Using graphical models also allows us to find the pose
of several targets

<div align='center'>
<img src='../assets/DeepCut.png' height=200>
<figcaption>
DeepCut
</figcaption>
</div>

Alternatively one ca do two stage process

1. Object Detection
2. Pose Estimation