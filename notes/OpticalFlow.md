# Optical Flow:
Optical flow refers to the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer (camera) and the scene. It helps understand how objects move in a sequence of images.

The optical flow problem involves estimating a dense vector field where each vector represents the displacement of a pixel from one frame to the next. This displacement field captures both the magnitude and direction of motion for each point in the image.

Key characteristics of optical flow:
1. **Dense Motion Field**: Unlike feature tracking which follows specific points, optical flow estimates motion for every pixel in the image
2. **Temporal Consistency**: Assumes that pixel intensities remain constant between consecutive frames (brightness constancy assumption)
3. **Spatial Smoothness**: Nearby pixels tend to move in similar ways (smoothness constraint)

# Methods:
- Patched Based
    1. Lucas-Kanade
    2. Horn-Shunck
- NN Based
    1. FlowNet

# Perception of Motion
<div center='align'>
<image src='../assets/percetionOfMotion.png'>
<figcaption>Figure 1: Illustration of optical flow and motion perception. The top left image shows a person running with a static camera (no camera motion), while the top right image shows the same person running but with the camera moving in the opposite direction. The bottom diagrams visualize the resulting optical flow vectors for each scenario.</figcaption>
</div>


**Assuming Image intensity is constant.**
<div center='align'>
<image src='../assets/opticalFlow.png'>
</div>

**Brightness Constancy Equation:**

$I(x,y,t) \approx I(x+dx,y+dy,t+dt)$

Using Taylor Series Expansion:


$I(x(t)+u.\Delta t,y+v.\Delta t) - I(x(t),y(t),t) \approx 0$

$I_x \cdot u +  I_y \cdot v + I_t = 0$ **(Brightness Constancy Constraint)**

$[u, v]$ is the optical flow.

## Lucas and Kanade

$E(u,v) = \int_{x,y} (I_xu+ I_yv+ I_t)^2 dxdy$

$\frac{\partial E(u, v)}{\partial u} = \frac{\partial E(u, v)}{\partial v}  = 0$

$2(I_xu+ I_yv+ I_t)I_x = 2(I_xu+ I_yv+ I_t)I_y = 0$

$\begin{bmatrix}
\sum I_{x}^2 & \sum I_{x}I_{y} \\
\sum I_{x}I_{y} & \sum I_{y}^2
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
= -
\begin{bmatrix}
\sum I_{x}I_{t} \\
\sum I_{y}I_{t}
\end{bmatrix}$

Structural Tensor representation:

$\begin{bmatrix}
T_{xx} & T_{xy} \\
T_{xy} & T_{yy}
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
= -
\begin{bmatrix}
T_{xt} \\
T_{yt}
\end{bmatrix}$


$u = \frac{T_{yt}T_{xy} - T_{xt}T_{yy}}{T_{xx}T_{yy} - T_{xy}^2} \text{ and } v = \frac{T_{xt}T_{xy} - T_{yt}T_{xx}}{T_{xx}T_{yy} - T_{xy}^2}
$


### Issues:
- Brightness constancy is not satisfied (Correlation based method could be used)
- A point may not move like its neighbors (Regularization based methods)
- The motion may not be small (Taylor does not hold!) (Multi-scale estimation could be used)

## Horn & Schunck
Global method with smoothness constraint to solve aperture problem

$E(u,v) = \int_{x,y} (I_xu+ I_yv+ I_t)^2 + \alpha^2(|\nabla u|^2 + |\nabla v|^2) dxdy$

$\frac{\partial E(u, v)}{\partial u} = \frac{\partial E(u, v)}{\partial v}  = 0$

$(I_xu+ I_yv+ I_t)I_x - \alpha^2(|\nabla u|)= (I_xu+ I_yv+ I_t)I_y + \alpha^2(|\nabla u|) = 0$


## FlowNet
End-to-end frame work to for optical flow prediction.
<div> <img src='../assets/FlowNet.png' width="700" height="250"> </div>

### SimpleNet
Both input images together and feed them through a rather generic network, allowing the network to decide itself how to process the image pair to extract the motion information.
<div> <img src='../assets/FlowNetSimple.png'> </div>

### FlowNetCorr
First produce meaningful representations of the two images separately and then combine them on a higher level. `correlation layer` performs multiplicative patch comparisons between two feature maps.

Correlation of 2 patches of size $K \times K$ is given by :

$c(\mathbf{x}_1, \mathbf{x}_2) = \sum_{\mathbf{o} \in [-k,k] \times [-k,k]} \langle \mathbf{f}_1(\mathbf{x}_1 + \mathbf{o}), \mathbf{f}_2(\mathbf{x}_2 + \mathbf{o}) \rangle$

where $K = 2k+1$. Above equation is similar to convolution but it is convolution of one data with another instead of filter.

<div> <img src='../assets/FlowNetCorr.png'> </div>

### FlowNetRefine
The main ingredient are ‘upconvolutional’ layers, consisting of unpooling (extending the feature maps, as opposed to pooling) and a convolution. To perform the refinement, we apply the ‘upconvolution’ to feature maps, and concatenate it with corresponding feature maps from the ’contractive’
part of the network and an upsampled coarser flow prediction (if available). 

This way we preserve both the high-level information passed from coarser feature maps and fine local information provided in lower layer feature maps.
<div> <img src='../assets/FlowNetRefine.png'> </div>

# Application
1. Motion based segmentation
2. SfM
3. Alignment (e.g., UAV analysis)
4. Video Compression
5. Object Tracking
6. Deformation Analysis
