# Object Tracking
To continuously locate and maintain the identity of target objects as they move through video frames.

Tracking is similarity measurement, correlation, correspondence, Matching/retrieval & data association.

# Learining

## Appearance
we need to know how the target looks like
- Single object tracking
- Re-identification

## Motion
To make predictions of where the targets goes
- Trajectory prediction


# Single Target Tracking

## As a matching/correspondence problem
- GOTURN: no online appearance modeling

**Input:** what to track?

**Architecture:** conv + concatenate + FC
<div>
<img src='../assets/goturn.png'>
</div>

**Pros:**
- No training, very fast as it is juct template matching problem

**Cons:**
- Does not work if the object moves very fast and goes out of search window.


## As an appearance learning problem
- MDNet: quick online finetuning of the network
- Slow: not suitable for real-time applications
- Solution: train as few layers as possible
<div>
<img src='../assets/MDNet.png'>
</div>

At test time, we need to train fc6 (up to fc4 if wanted)
<div>
<img src='../assets/MDNetTesttime.png'>
</div>

<div>
<img src='../assets/MDNetOnline.png'>
</div>

**Pros:**
- No previous location assumption, the object can move
anywhere in the image

**Cons:**
- Not as fast as GOTURN


## As a (temporal) prediction problem
- ROLO = CNN + LSTM
<div>
<img src='../assets/ROLO.png'>
</div>
LSTM receives the heatmap for the object’s position
and the 4096 descriptor of the image
<div>
<img src='../assets/ROLOArchitecture.png'>
</div>



## Challanges
- Occlusions
- Viewpoint/pose/blur/illumination variations (in a few frames of a sequence)
- Background clutter


# Multiple Object Tracing

## Online Tracking
- Processes two frames at a time
- For real-time applications
- Prone to drifting à hard to recover from errors or occlusions

**Process**
- Track initialization (e.g. using a detector)
- Prediction of the next position (motion model)
    - [Kalman filter](KalmanFilter.md)
    - Recurrent architecture
    - constant velocity model (works really well at high
framerates and without occlusions!)

- Matching predictions with detections (appearance model)
    - [Bipartite matching](BipartiteMatching.md)


### Track initialization
Making a detector into a tracktor
- Tracktor: a method trained as a detector but with tracking capabilities.
- Where did the detection with ID1 go in the next frame? 

**Two-Step Detector**
- Region Proposal
- Regression

**Pros**
- Tracktor are online
- We can train our model on still images
- We can reuse an extremely well-trained regressor

**Cons**
- Confusion in crowded places as there is no notion of identification.
- The track is killed if the target becomes occluded.
- Will not work if the object or the camera has large motions.

1st & 2nd can be solved using ReID (Re-Identification). While 2nd & 3rd can be solved using Motion model.

Modeling Appearence : Re-ID
Modeling Motion : Model Motion 

### Prediction of the next position

### Matching predictions with detections (appearance model)



## Offline Tracking
- Processes a batch of frames
- Good to recover from occlusions
- Not suitable for real-time applications
- Suitable for video analysis


## Challanges
- Multiple objects of the same type
- Heavy occlusions
- Appearance is often very similar

# Applications
- Surveillance & Security
- Traffic Monitoring
- Autonomous Vehicles
- Sports Analytics
- Human-Computer Interaction
- Medical Imaging
- Robotics