# Storm-damage-assesment

Introduction
Hurricane Maria was one of the most devastating natural disasters to hit the Caribbean, leaving a trail of destruction in its wake. Accurate and rapid assessment of building damage in the aftermath is crucial for effective response and recovery efforts. This project, developed as part of the EY Data Challenge, leverages advanced artificial intelligence (AI) and machine learning (ML) techniques to automatically distinguish between damaged and undamaged residential and commercial buildings using aerial imagery captured post-Hurricane Maria. Our solution aims to provide emergency responders, urban planners, and humanitarian organizations with critical information to prioritize aid, allocate resources efficiently, and plan long-term rebuilding strategies.

For train.py:
To train a Faster R-CNN model on your dataset, use the following command:
```
python train.py --data /path/to/dataset --output /path/to/save/model --num_workers 4
```

This command initiates the training process for a Faster R-CNN model using the specified dataset. The trained model will be saved to the provided output path. Adjust the --num_workers parameter according to your system's capabilities for optimal data loading performance.

For eval.py:
To evaluate a model on a set of images, use the following command:

```
python eval.py --input_dir /directory/of/input/images --output_dir /directory/to/save/output/images --chkpt /path/to/model/checkpoint --nms True
```

This command evaluates a trained model using images from the specified input directory, saves the output images to the output directory, and uses the provided model checkpoint. The --nms flag enables or disables non-maximum suppression based on the provided boolean value.

Project Directory Structure
This section outlines the structure of the project directory, providing a clear overview of where to find specific scripts, datasets, and output directories. Each component of the project is organized into specific folders for ease of access and management.

```
storm-damage/
│
├── data/ # Directory containing the dataset for training and evaluation
│ ├── building_footprints/ # building_footprints from EY
│ ├── test/ # Submission images from EY
│ ├── roboflow/ # datasets downloaded from roboflow
│ ├── intermediate/ # processed images that are then annotated in roboflow
│ ├── eval/ # annotations of images (output of eval.py)
│ └── raw/ # raw data from EY

├── EDA/ # Contains saved models and checkpoints
│ ├── tile_generation.ipynb/ # deterministic tile data generation filtered for buildings
│ └── ... # Other EY notebooks
│
├── utils/ # utils for training
│ └── data.py/ # dataset class for Pytorch for dealing with YOLOv8 format
│
├── weights/ # Output directory for model weights
│
├── xBD_utils/ # utils for working with xBD_dataset
│
├── xView2_baseline/ # git repo for example usuage of xBD dataset
│
├── train.py # training file for FasterRCNN
│
├── eval.py # evaluation file for FasterRCNN
│
└── README.md # Documentation explaining the project's purpose, structure, and usage
```

# Tips + Tricks
1. Download the dataset from Roboflow in YoloV8 format and save it to ./data/roboflow
