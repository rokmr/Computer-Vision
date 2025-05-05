# Object Detection
The goal of object detection is to predict a set of bounding boxes(x,y,w,h) and category labels for each object of interest.

<div>
<img src='../assets/ObjectDetection.png'>
</div>

# Traditional
## Template Matching + Sliding Window
For every position you evaluate how much do the pixels in the image and template correlate.

**Cons**

1. Does not handle occlusions.
2. Works with instance of object but not with class of it.
3. Does not work if pose changes.
4. Does not work if position, scale and aspect ratio changes.

## Feature Extraction and Classification
Learn multiple weak classifier to build a strong final decision.

### Feature Extraction
**Viola-Jones Detector**

Haar Features
<div>
<img src='https://upload.wikimedia.org/wikipedia/commons/8/8a/Haar_Feature_that_looks_similar_to_the_bridge_of_the_nose_is_applied_onto_the_face.jpg'>
<img src='https://upload.wikimedia.org/wikipedia/commons/6/69/Haar_Feature_that_looks_similar_to_the_eye_region_which_is_darker_than_the_upper_cheeks_is_applied_onto_a_face.jpg'>
</div>

**Histogram of Oriented Gradients(HOGs)**
Compute gradients in dense grids, compute gradients and create a
histogram based on gradient direction
<div>
<img src='https://scikit-image.org/docs/0.25.x/_images/sphx_glr_plot_hog_001.png'>
</div>

### Classification
It is done with the help of SVM.


## General Object Detection

- Class agnostic
- Object Proposals / Region of Intrest
    - [Selective search](detection/SelectiveSearch.md)
    - Edge boxes


### Non-Maxium Supression (NMS)
<div>
<img src='https://thepythoncode.com/media/articles/non-maximum-suppression-using-opencv-in-python/non-max-suppression.webp'>
</div>

<div>
<img src='../assets/NMSAlgo.png'>
</div>

#### IoU
If two bbox ahve both Intersection and union high then the IoU will be high.
Assume two case :

1. Intersion is large but the bbox are also very large bbox this leads to very high union depicting same same object.
2. Intersion is large but the bbox are small bbox then possibly they are dipicting different object.

<div>
<img src='../assets/IoU.png'>
</div>

#### [$\lambda_{NMS}$](https://arxiv.org/pdf/1511.06437)
Do not allow the bbos if they are overlapping more than $\lambda_{NMS}$ threshold.
<div>
<img src='../assets/NMSIssue.png'>
</div>

Narrrow Threshold (High IoU) : Low Precision (More False Positive)
Wide Threshold (Low IoU): Low Recall (More False Negative)

#### Localization

### Two-Stage Detector

- R-CNN, Fast R-CNN, Faster R-CNN
- SPP-Net, R-FCN, FPN

<div>
<img src='../assets/TwoStageDetector.png'>
</div>

- [Overfeat](detection/Overfeat.md)
- [RCNN](detection/RCNN.md)

### One-Stage Detector

- YOLO, SSD, RetinaNet
- CenterNet, CornerNet, ExtremeNet

<div>
<img src='../assets/OneStageDetector.png'>
</div>





# Methods

- [SSD](detection/SSD.md)
- [YOLO](detection/YOLO.md)
- [RetinaNet](detection/RetinaNet.md)
- [DETR](detection/DETR.md)
- [Swin Transformer](detection/SwinTransformer.md)
- [DINO](detection/DINO.md)
- [InternImage](detection/InternImage.md)
- [OWL](detection/OWL.md)