# Semantic Predictive Control for Explainable and Efficient Policy Learning
<p align="center"><img width="70%" src="assets/carla_pred.png" /></p>

[[paper]](https://go.yf.io/spc-paper) / [[video demo]](https://youtu.be/FSrzyR8UhxM)

Semantic predictive control (SPC) is a policy learning framework that predicts future semantic segmentation and events by aggregating multi-scale feature maps.
It utilizes dense supervision from semantic segmentation for feature learning and greatly improves policy learning efficiency. 
The learned features are explainable as they depict future scenes with semantic segmentation and explicit events.

This repository contains a [PyTorch](https://pytorch.org/) implementation of SPC, as well as some training scripts to reproduce policy learning results reported in our paper.

- [Overview](#overview)
- [Training](#training)

## Overview
<p align="center"><img width="70%" src="assets/SPN_overview.png" /></p>
Our model is composed of four sub-modules: 

1. The feature extraction module extracts multi-scale intermediate features from RGB observations;
2. The extracted features are then concatenated with tiled actions and feed to the multi-scale prediction module that sequentially predicts future features; 
3. The information prediction module takes in the predicted latent feature representation and outputs corresponding future frame semantic segmentation and task-related signals, such as collision, off-road, and speed;
4. The guidance network module that predicts action distribution for efficient sampling-based optimization.  

## Training

Our results in the paper can be reproduced with the provided scripts by running
```bash
cd scripts/
bash train_#ENVNAME.sh
```
Together with the training scripts, simulator environments need to be activated:

### Carla

To train on carla, the [carla simulator](http://carla.org/) should be started at first, we give an example with default settings here:

```
### On Ubuntu
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh -carla-settings=Example.CarlaSettings.ini -windowed -ResX=256 -ResY=256 -carla-server -carla-no-hud

### On Windows
CarlaUE4.exe -windowed -ResX=800 -ResY=600 -carla-server -carla-no-hud -carla-settings=Example.CarlaSettings.ini
```
By default, the message port of Carla Simulator is 2000 and the **--port** arg should be set as the same.

## Evaluation
To evaluate the model and to produce demo with the latest model saved, we can simply run *main.py* with the flag **--eval**. Then, to transform saved snapshot images to a demo video, simply run the script:

```
python merge_demo.py
```
