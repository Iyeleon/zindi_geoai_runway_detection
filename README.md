# ZINDI GEOAI Clandenstine Runway Detection Competition
This repository contains the code for detecting clandestine runways in Peru, developed as part of the GEOAI Runway Detection competition hosted by Zindi. 
The project tackles the challenge as both a localization and segmentation problem using satellite imagery and deep learning techniques. 
The dual model detects if a runway is present in an image and then segments the runway. 


## Methodology Used
### Dataset collection
The dataset is open-source satellite imagery from Sentinel-2 at a 10m resolution. The dataset collection process involved several key steps:
- Buffering and rasterizing runway linestrings in the provided train csv files to create segmentation masks and define bounds for downloading corresponding satellite imagery.
- Extract satellite imagery corresponding to the rasterized locations, ensuring to gather data from both runway areas and adjacent regions without runways to create a balanced dataset.
- Selection of Sentinel-2 bands that highlighted features indicative of runways - Red, Blue, Green, NIR and SWIR bands.  

Each satellite imagery was downloaded from the specific year of detection provided in the competition, under the assumption that all runways in their year of observation should be active. 
This yielded approximately 150 active runway and 150 non-runway images for training and evaluation. 

### Model:
The model leverages Segformer, a transformer-based architecture for segmentation, combined with a separate classification head that determines if an image contains a runway.

## Getting Started
Follow the instructions below to set up and run the project:
### Prerequisites
- python >= 3.12
- Tensorflow[and-cuda] == 2.17.0
- transformers==4.43
- LinuxOS
- Openeo oidc Authentication

This notebook requires a GPU (with 8GB RAM) to run successfully. With fewer resources, change segformer config in config.toml from b3 to a smaller variant.
Ensure you have the necessary dependencies installed. You can install them using pip or by following the setup script provided.
```
conda env create -f environment.yml
conda activate zindi_geoai
```

### Clone this repository
- Clone with https
```
git clone https://github.com/Iyeleon/zindi_geoai_runway_detection.git
```
- Clone with ssh
```
git clone git@github.com:Iyeleon/zindi_geoai_runway_detection.git
```

## How to Run
1. Change directory into the repository folder
```
cd zindi_geoai_runway_detection
```

2. Run `make` commands to download the data. Downloading the train or test dataset invokes OIDC authentication if user is not already authenticated. It will display a link in the terminal. You can copy and paste this link into a web browser on any device to complete the authentication. 

*Please note that openeo has monthly download limits.*
- Make Training Data: Generate the dataset to be used for training the model.
```
make train_data
```
- Make Test Data: Prepare the test dataset for inference and submission.
```
make test_data
```
- Run Model: Train the Segformer model using the prepared training data.
```
make run_notebook
```

## Other Explorations
In addition to the Segformer-based approach, I explored using YOLO (You Only Look Once) for object detection in runway localization. To support this, I created a [YOLO segmentation dataset](https://www.kaggle.com/api/v1/datasets/download/iyeleon/clandenstine-runways-yolo-segmentation-dataset) specific to the task. YOLO produced impressive results during training and validation, showing a high detection rate and accurate bounding box placement for runway regions. However, it faced challenges generalizing to the test set due to the limited dataset size, leading to overfitting. While the YOLO model demonstrated promise, its performance indicated a need for a larger and more diverse training set to improve robustness across unseen data.

You can find the work on YOLO in the dedicated [yolo_exploration](https://github.com/Iyeleon/zindi_geoai_runway_detection/tree/yolo_exploration) branch in this repository.

## Future Work
Building on the current model's success and the insights gained from YOLO, there are several promising directions for future work:

Expand the Dataset: To further enhance YOLO's performance, a larger dataset with greater diversity in runway types, environmental conditions, and geographical locations would help improve generalization and reduce overfitting.
Experiment with SAM2: The Segment Anything Model (SAM2) with bounds and mask prompts could provide more refined segmentation results. SAM2's ability to incorporate flexible input prompts may enable it to focus on specific regions within complex backgrounds, potentially improving accuracy in detecting and segmenting runway areas in varied contexts.
This future work will focus on creating a more comprehensive model capable of robust, high-precision runway detection across diverse conditions and landscapes.