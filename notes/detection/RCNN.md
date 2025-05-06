# Content
1. [RCNN](#rcnn)
2. [SPP-Net](#spp-net)
3. [Fast RCNN](#fast-rcnn)
4. [Faster RCNN](#faster-rcnn)
5. [Conclusion](#conclusion)

# [RCNN](https://arxiv.org/pdf/1311.2524)

<div>
<img src='../../assets/RCNNOverview.png'>
</div>

Steps

1. Scan the input image for possible objects using an algorithm called Selective Search, generating ~2000 region proposals

2. Warp to a fix size 227 x 227

3. Run a convolutional neural net (CNN) on top of each of these region proposals

4. Make the output of each CNN and feed it into:

     a) an SVM to classify the region and 
     
     b) a linear regressor to tighten the bounding box of the object, if such an object exists.

<div>
<img src='../../assets/RCNN.png'>
</div>


## Training
1. Pre-train the CNN on ImageNet
2. Finetune the CNN on the number of classes the detector is aiming to classify (softmax loss).
3. Train a linear Support Vector Machine classifier to classify image regions. One SVM per class! (hinge loss)
4. Train the bounding box regressor (L2 loss)

**Cons**
1. If we have overlapping window then we will do ConvNet computation for each of the pixels more than 1 times. This increases extra computation.

2. Training is slow and complex(no end-to-end)

3. Region Proposal region is fixed


# [SPP Net](https://arxiv.org/pdf/1406.4729)
<div>
<img src='../../assets/SPPNetOverview.png'>
</div>

Makes the RCNN fast at test time.

<div>
<img src='../../assets/SPPNet.png'>
</div>

**Issues**

1. Training is slow and complex(no end-to-end)

2. Region Proposal region is fixed

# [Fast RCNN](https://arxiv.org/pdf/1504.08083)

<div>
<img src='../../assets/FastRCNNOverview.png'>
</div>
<div>
<img src='../../assets/FastRCNN.png'>
</div>

1. Performing feature extraction over the image before proposing regions, thus only running one CNN over the entire image instead of 2000 CNN’s over 2000 overlapping regions.
2. After conv5 there is FC layer we need to make all the deature size need to be of same size using RoI Pooling layer.
3. Replacing the SVM with a softmax layer, thus extending the neural network for predictions instead of creating a new model

<div  align='center'>
<img src='../../assets/RoIPoolingLayer.png'  height=300 width= 370>
</div>

# [Faster RCNN](https://arxiv.org/pdf/1506.01497)
- Removes region proposal network
Can we reuse our CNN feature and still be able to create this proposal.

<div align='center'>
<img src='../../assets/FasterRCNN.png' height=200 width= 200>
<img src='../../assets/FasterRCNNRPN.png' height=200 width= 300>
</div>

How to extract proposals.

- How many proposals?
     - Decide a fix number
     - set of 9 anchor box per location (3 scales, 3 aspect ratio)

- Where are they placed?
    - Densly

1. For each of the location get the descriptor of 256-d by 3X3 filtermap

2. Pass the descriptor to the classification layer and regression layer



## RPN Training
<div align='center'>
<img src='../../assets/FasterRCNNRPNExplained.png'>
<figcaption>
Region Proposal Network
</figcaption>
</div>

**Classification Ground-Truth**

$p^{*}$ Amount of anchor box overlapping with the Ground-Truth.

$p^{*} = 1$ if IoU > 0.7 (anchor is in foreground)

$p^{*} = 0$ if IoU < 0.3 (anchor is in background)

For training only consider above two case.


1. Randomly sample 256 sample, form mini-batch.
2. Calculate binary CE loss
3. Anchor box containing object will go through regression box

Anchor box $(x_a, y_a, w_a, h_a)$, $x_a, y_a$ is center of box and rest width and height respectively.

4. Network actually predicts are $(t_x, t_y, t_w, t_h)$ which are relative.

$t_x = (x-x_a)/w_a$

$t_y = (y-y_a)/h_a$

$t_w = \log(w/w_a)$

$t_h = \log(h/h_a)$

5. Smooth L1 loss on regression targets

## Faster RCNN Training
Can be train jointly. But in paper it is trained in following manner.
    - RPN classification (object/non-object)
    - RPN regression (anchor -> proposal)
    - Fast R-CNN classification (type of object)
    - Fast R-CNN regression (proposal -> box)

<div align='center'>
<img src='../../assets/FasterRCNNTraining.png'>
</div>

**Pros**

1. 10x faster at test time wrt Fast R-CNN
2. Trained end-to-end including feature extraction, region proposals, classifier and regressor.
3. More accurate, since proposals are learned. RPN is fully convolutional.

# Conclusion

|                           | R-CNN | Fast RCNN | Faster RCNN |
|---------------------------|-------|-----------|-------------|
| Test time per image (sec) | 50    | 2         | 0.2         |
| Speeed-Up                 | 1X    | 25X       | 250X        |
| mAP (VOC 2007)            | 66.0  | 66.9      | 66.9        |