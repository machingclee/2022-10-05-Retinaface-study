# Main Referenece

- https://github.com/biubug6/Pytorch_Retinaface

# Changes

- The original project only support facial landmarks of 10 points. Code modification has been made to support an arbitrary number of points, setting can be found in `src.config.n_landmarks`.
- Some additional work has been made to part of the code that is hard-coded 5 times for 5 landmarks (10 poinsts) which cannot be generalized.
- Due to structural difference of annotation file:
  - New `Dataset` object in `data/wflw.py`.
  - New `collate_fn` for `Dataloader`.
  - Training script has been kept minimal level of difference (almost same as before).
- Network modification has been made before feeding features into different detection heads (`ClassHead`, `BboxHead`, `LandmarkHead`, etc).
- Attention mechanism has been added by introducing Squeeze and Excitation Block (SEBlock) after each feature of different scales after `SSH`'s. It mainly learns which channels are relatively more important for different detection.
- The training is done using **W**ider **F**acial **L**andmarks in the **W**ild (WFLW) dataset.

# Dataset Check of WFLW

Each of the ground truths in WFLW look:

<img src="dataset_check/039.jpg" width="400"/>

# Why WFLW Dataset?

I want the relative location of pupil for further study and WFLW is the one that also annotate pupil.

# Sample Result

## When Using Landmarks of 10 Points (Wider Face)

<img src="images_for_readme/001.png"/>

## When Using Landmarks of 196 Points (WFLW) with SEBlocks

Still training in progress, different hyper-parameters are still being investigated

At epoch 41:

- green box = ground truth
- blue box = predictions

Prediction threshold has be set to 0.1 only. Higher threshold can rule out inappropriate boxes.

<img src="images_for_readme/epoch_041_batch_00200.jpg" width="350"/>
<img src="images_for_readme/epoch_041_batch_00300.jpg" width="350"/>
<img src="images_for_readme/epoch_041_batch_00380.jpg" width="350"/>
<img src="images_for_readme/epoch_041_batch_00130.jpg" width="350"/>
