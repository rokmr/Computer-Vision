# Content
1. [YOLO](#yolo)
2. [YOLO-World](YOLO-World)
3. [YOLO Paper](#yolo-papers)


# YOLO
It does not have region proposal network and also it does not have the fully-connected layer. 

<div align='center'>
<img src='../../assets/YOLO_img1.png'>
<figcaption>
You Only Look Once
</figcaption>
</div>

## Process

1. Divide the image into grid (SxS cells).
2. Predict B anchor box at the center of each box along with the confidence score
3. Predict C classes for each grid cell.

YOLO-Tensor : SxS(Bx5 + C) 

where,
- SxS : number of grid

- B : Number of bbox $(P_c, b_x, b_y, b_h, b_w)$

- C : Number of classes

<div align='center'>
<img src='../../assets/YOLO_img2.png'>
<figcaption>
Predicting bounding box and confidence score for each cell.
</figcaption>
</div>



<div align='center'>
<img src='../../assets/YOLO_img3.png'>
<figcaption>
Predicting class probability for each cell.
</figcaption>
</div>

## Loss Function

$L_{total} = L_{localization} + L_{confidence} + L_{classification}$

$ L_{localization} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]$

$L_{confidence} =  \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2$

$L_{classification} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c\in classes} (p_i(c) - \hat{p}_i(c))^2$

$\lambda_{coord} = 5$

$\lambda_{noobj} = 0.5$

<div align='center'>
<img src='../../assets/YOLO_Loss.png' height =250 width = 300>
</div>


# YOLO Papers

#### [YOLO Survey](https://arxiv.org/html/2304.00501v6)
#### [YOLO-v1](https://arxiv.org/pdf/1506.02640) || [Slides](https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/edit?slide=id.g15092aa245_0_155#slide=id.g15092aa245_0_155)

#### [YOLO-v2](https://arxiv.org/pdf/1612.08242) || [Slides](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit?slide=id.p#slide=id.p)

#### [YOLO-v3](https://arxiv.org/pdf/1804.02767)

#### [YOLO-v4](https://arxiv.org/pdf/2004.10934)

#### [YOLO-v5](https://github.com/ultralytics/yolov5) || [colab](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/ultralytics_yolov5.ipynb)

#### [YOLO-v6](https://arxiv.org/pdf/2209.02976) || [colab](https://github.com/meituan/YOLOv6/blob/main/inference.ipynb)

#### [YOLO-v7](https://arxiv.org/pdf/2207.02696) || [colab](https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb)

#### [YOLO-v8]()

#### [YOLO-NAS]()