# Bipartite Matching

<div>
<img src='../assets/BipartiteMatching1.png' height= '200px'>
<img src='../assets/BipartiteMatching2.png' height= '200px'>
</div>



- Links detections with predictions using distance metrics
- Uses IoU, Pixel, or 3D distances between boxes

## Process
- Calculate distances between boxes
- Apply Hungarian algorithm for optimal matching
- Set thresholds to handle missing/unsuitable matches

## Edge Cases
- Missing prediction: Add dummy nodes
- No suitable match: Use cost threshold
- Unmatched boxes: Create new tracks

## Output
- Matched pairs
- New tracks from unmatched detections
- Lost tracks from unmatched predictions

## Hungarian algorithm
[Demo](http://www.hungarianalgorithm.com/solve.php)

<div align="center">
   <iframe src="https://drive.google.com/file/d/1xDCm92v2X-OB65SWoyp_j5eFre2Hhy5s/preview" width="640" height="480" allow="autoplay"></iframe>
</div>
