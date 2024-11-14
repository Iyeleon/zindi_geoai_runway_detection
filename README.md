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
Ensure you have the necessary dependencies installed. You can install them using pip or by following the setup script provided.
```
conda env create -f environment.yml
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

### How to Run
Change directory into the repository folder
```
cd zindi_geoai_runway_detection
```

Run `make` commands to download the data, run the model and generate predictions
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