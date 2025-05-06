# [Overfeat](https://arxiv.org/pdf/1312.6229)

Slidingwindow + bbox regression + classification

## Sliding Window
Implicity encoded in the CNN architecture. Use sliding widow at different scale.

## Localization
Regression

## Detection
Classification


## Cons
- Needs fixed sized window as the fully-connected layer need to have fixed input.
- Expensive to try out all the possible positions, scales and aspect ratio. (Choose only the potential location)




