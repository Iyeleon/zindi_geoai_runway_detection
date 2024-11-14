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

Segformer stands out for its ability to efficiently capture both local and global contexts in an image, making it well-suited for high-resolution remote sensing tasks like runway detection.

#### Segformer Backbone:

- Transformers for Vision: Segformer uses Vision Transformers (ViTs) for feature extraction, where each image is processed as a series of patches. This design allows it to capture long-range dependencies, providing better context and clarity when differentiating runways from surrounding features.
- Efficient Patch Merging: Unlike traditional convolutional neural networks (CNNs), Segformer performs hierarchical patch merging to maintain efficiency while scaling up receptive fields. This feature is particularly valuable for images where clear boundaries and fine details, like runways, are crucial.
- Multi-Scale Feature Aggregation: The model aggregates features at multiple scales, allowing it to focus on both fine-grained details and broader contextual information, which is crucial for identifying runways among surrounding terrain.
#### Classification Head
- In addition to the segmentation backbone, the model includes a classification head that determines if a runway is present in an image.  
- This head serves as a binary classifier that flags the presence or absence of a runway based on global image features learned through transfer learning.  

The combined setup of segmentation and classification improves accuracy by allowing the model to handle both localization (precisely identifying the runway region) and detection (confirming the presence of a runway).
#### Training:

- The model is a single model with two outputs. The model is trained on 512 x 512 images from Sentinel-2, labeled as either containing active runways or not. Each image has a resolution of 10m, which provides sufficient detail to distinguish runways from other landscape elements like roads, rivers, or empty fields.
- Data augmentation techniques, including rotation, flipping, and color adjustments, were applied to make the model robust to variations in runway orientation and lighting conditions.
- Additional data augmentation includes dynamically generating synthetic road paths to help the model distinguish them from runways. 

#### Results
Performance Metrics:

- Intersection over Union (IoU): The model achieved high IoU scores on runways, which indicates precise segmentation and minimal overlap with non-runway areas.

- Accuracy of Runway Detection: The separate classification head allowed for a reliable accuracy rate in detecting images with runways, ensuring that few images without runways were misclassified as containing runways. Validation accuracy ranges from 0.88 - 0.94

Qualitative Observations:

- The model consistently detected and segmented runways across various geographical and environmental conditions, including runways located in diverse terrains.
- It demonstrated robustness to different runway orientations and surface types, providing reliable segmentation even when the runway was partially obstructed by objects like aircraft or vehicles.
- False Positives in pathways

### Inference


### Post Processing


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
## Limitations
Despite the strong performance, there are limitations and areas for improvement:

- Cloud Cover and Atmospheric Conditions: The model struggles with images that have significant cloud cover or haze, which can obscure runways. Clouds introduce noise in segmentation results, sometimes leading to false positives or negatives.
While cloud masks were applied to mitigate this, they were not always effective in fully eliminating the impact of cloud-covered regions.
Complex Backgrounds and Water Bodies:

- Runways located near water bodies, rivers, or roads sometimes lead to confusion. Although the model incorporates non-RGB bands like NDWI, distinguishing between certain water patterns and runways remains challenging.
Further enhancement of the input feature set or use of additional bands could help improve differentiation in these scenarios.

- Resolution Limitations: The 10m resolution from Sentinel-2 is adequate but not ideal for extremely fine-grained segmentation. Higher-resolution imagery could potentially yield better results, especially for smaller or less distinct runways.
The use of 10m data may also contribute to boundary blurring around runways, affecting the accuracy of pixel-level segmentation.
- Computational Resource Constraints: 
Transformer-based models like Segformer can be computationally intensive. For larger datasets or higher resolutions, the model may face memory and processing limitations, especially on GPUs with restricted VRAM.
During training, memory management and batch size adjustments were essential to avoid out-of-memory errors, which could impact training efficiency.
- Dataset Limitations: Since the dataset assumes that all runways in the year of observation are active, there could be inaccuracies if certain runways were inactive but still visible in the imagery. This limitation introduces potential false positives in the classification task.
Future Directions

To address these limitations, potential future improvements could include:

- Integrating imagery from other satellite sources.
- Engineering band features to improve performance
- Using advanced cloud masking or image enhancement techniques to reduce the impact of cloud cover.
- Exploring multi-modal models that incorporate both optical and radar data to improve performance in challenging conditions.
- Optimizing the model architecture to balance accuracy with computational efficiency.

## Other Explorations (During the Competition)
In addition to the Segformer-based approach, I explored using YOLO (You Only Look Once) for object detection and segmentation. The yolo segmentation model performs localization and segmentation, which is the objective of my approach. To support this, I created a [YOLO segmentation dataset](https://www.kaggle.com/api/v1/datasets/download/iyeleon/clandenstine-runways-yolo-segmentation-dataset) specific to the task. YOLO produced impressive results during training and validation, showing a high detection rate and accurate bounding box placement for runway regions. However, it faced challenges generalizing to the test set due to the limited dataset size, leading to overfitting. While the YOLO model demonstrated promise, its performance indicated a need for a larger and more diverse training set to improve robustness across unseen data.

You can find the work on YOLO in the dedicated [yolo_exploration](https://github.com/Iyeleon/zindi_geoai_runway_detection/tree/yolo_exploration) branch in this repository.

## Future Work
Building on the current model's success and the insights gained from YOLO, there are several promising directions for future work:

- Expand the Dataset: To further enhance YOLO's performance, a larger dataset with greater diversity in runway types, environmental conditions, and geographical locations would help improve generalization and reduce overfitting.

- Experiment with YOLO, Localization and SAM2: The Segment Anything Model (SAM2) with bounds and mask prompts could provide more refined segmentation results. SAM2's ability to incorporate flexible input prompts may enable it to focus on specific regions within complex backgrounds, potentially improving accuracy in detecting and segmenting runway areas in varied contexts.

This future work will focus on creating a more comprehensive model capable of robust, high-precision runway detection across diverse conditions and landscapes.