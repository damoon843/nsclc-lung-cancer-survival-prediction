# Applications of 3D-CNN in Lung Cancer Patient Survival Prediction
Brown University Statistics Capstone Project 

**Please refer to [the poster](https://github.com/damoon843/nsclc-lung-cancer-survival-prediction/blob/main/stats_capstone.pdf) for a more in-depth overview of this research.**

## Purpose
We develop a convolutional neural network (CNN) for survival analysis of patients with non-small cell lung cancer (NSCLC). This research is part of a comparative study (currently unpublished) that examines survival prediction performance across the following methods:
- Cox proportional hazard model 
- Random survival forests
- 2D CNN (at the slice level) with transfer learning
- 3D CNN (the current research)

## Data 
The [dataset](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics) contains images from 422 non-small cell lung cancer (NSCLC) patients. In addition, clinical data and manual delineation by a radiation oncologist of the tumor volume are used to extract radiomic features.

