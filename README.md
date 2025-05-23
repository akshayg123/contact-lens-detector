# Contact Lens Defect Detection System

An AI-powered web application that detects defects in contact lenses using computer vision and deep learning.

## Overview

This system uses a custom YOLO-style neural network to detect and classify defects in contact lens images. It provides both a command-line interface for training and a web interface for real-time defect detection.

## Features

- **Real-time Defect Detection**: Upload images through a web interface for immediate defect analysis
- **Custom YOLO Architecture**: Specialized neural network for precise defect localization
- **Web Interface**: User-friendly Flask web application for easy interaction
- **Multi-class Detection**: Capable of identifying multiple types of contact lens defects
- **Visual Results**: Displays detected defects with bounding boxes and confidence scores

## Requirements

```txt
tensorflow
opencv-python
numpy
flask
matplotlib
pandas
```

## Project Structure

```
contact_lens_project/
├── app.py                  # Web application server
├── train_detector.py       # Model training script
├── predict_image.py        # CLI prediction script
├── model_config.json       # Model configuration
├── templates/             
│   ├── index.html         # Upload page
│   └── result.html        # Results display page
└── dataset_images/        # Training dataset directory
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd contact_lens_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Place your labeled dataset in the `dataset_images/` directory
2. Ensure your annotations are in `_annotations.csv`
3. Run the training script:
```bash
python train_detector.py
```

### Running the Web Interface

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image through the web interface to get defect detection results

## Model Architecture

The system uses a custom YOLO-style architecture with:
- Input image size: Configurable (default in model_config.json)
- Grid-based detection system
- Custom loss function for object detection
- Non-max suppression for overlapping detection cleanup

## Training Data

The model expects training data in the following format:
- Images in `dataset_images/` directory
- Annotations in CSV format with columns:
  - filename
  - class
  - xmin, ymin, xmax, ymax (normalized coordinates)
  - width, height



## Acknowledgments

- Based on YOLO architecture principles
- Implemented using TensorFlow and Keras
- Web interface powered by Flask