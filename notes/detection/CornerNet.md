# [CornerNet](https://arxiv.org/pdf/1808.01244)

Bounding box cordinates as top-left and bottom-right corner.

<div align='center'>
<img src='../../assets/CornerNet_img1.png' width =600>
</div>

<div align='center'>
<img src='../../assets/CornerNet_img3.png' width =600>
<figcaption>
Hourglass network
</figcaption>
</div>

<div align='center'>
<img src='../../assets/CornerNet_img2.png' width =600>
<figcaption>
Corner pooling
</figcaption>
</div>


## Issues
- Many incorrect bounding boxes (especially small) $\rightarrow$
too many False Positives
- Hypothesis: It is hard to infer the class of the box if the
network is focused on the boundaries