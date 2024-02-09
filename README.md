## Storm-damage-assesment

Introduction
Hurricane Maria was one of the most devastating natural disasters to hit the Caribbean, leaving a trail of destruction in its wake. Accurate and rapid assessment of building damage in the aftermath is crucial for effective response and recovery efforts. This project, developed as part of the EY Data Challenge, leverages advanced artificial intelligence (AI) and machine learning (ML) techniques to automatically distinguish between damaged and undamaged residential and commercial buildings using aerial imagery captured post-Hurricane Maria. Our solution aims to provide emergency responders, urban planners, and humanitarian organizations with critical information to prioritize aid, allocate resources efficiently, and plan long-term rebuilding strategies.

For train.py:
To train a Faster R-CNN model on your dataset, use the following command:
`python train.py --data /path/to/dataset --output /path/to/save/model --num_workers 4`

This command initiates the training process for a Faster R-CNN model using the specified dataset. The trained model will be saved to the provided output path. Adjust the --num_workers parameter according to your system's capabilities for optimal data loading performance.

For eval.py:
To evaluate a model on a set of images, use the following command:

`python eval.py --input_dir /directory/of/input/images --output_dir /directory/to/save/output/images --chkpt /path/to/model/checkpoint --nms True`

This command evaluates a trained model using images from the specified input directory, saves the output images to the output directory, and uses the provided model checkpoint. The --nms flag enables or disables non-maximum suppression based on the provided boolean value.

Project Directory Structure
This section outlines the structure of the project directory, providing a clear overview of where to find specific scripts, datasets, and output directories. Each component of the project is organized into specific folders for ease of access and management.

```
EY-Hurricane-Maria-Damage-Assessment/
│
├── data/ # Directory containing the dataset for training and evaluation
│ ├── train/ # Training dataset with annotated images
│ └── test/ # Test dataset for evaluating the model
│
├── models/ # Contains saved models and checkpoints
│ ├── faster_rcnn/ # Saved Faster R-CNN model checkpoints
│ └── ... # Other model checkpoints (if any)
│
├── src/ # Source code for the project
│ ├── train.py # Script for training the model
│ ├── eval.py # Script for evaluating the model
│ └── utils/ # Utility scripts and modules
│
├── outputs/ # Output directory for evaluation scripts
│ ├── detected_images/ # Images with damage detections
│ └── reports/ # Evaluation reports and metrics
│
├── notebooks/ # Jupyter notebooks for experiments and analyses
│
├── requirements.txt # The required libraries and dependencies for the project
│
└── README.md # Documentation explaining the project's purpose, structure, and usage
```
