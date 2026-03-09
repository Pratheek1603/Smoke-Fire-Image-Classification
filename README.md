Smoke / Fire Classification using OpenCV Neural Network (C++)

This project trains a neural network in C++ using OpenCV's ANN_MLP module to classify smoke and fire from image regions.

The dataset contains 14,000+ images with YOLO-format bounding box annotations.

The program:

extracts object regions from images

preprocesses them

converts them into feature vectors

trains a fully connected neural network for binary classification

The trained model is saved as:

smoke_fire_nn.xml

This model can later be loaded for inference.

Project Overview

Main steps in the project:

Load images and YOLO annotations

Extract labeled regions from images

Resize regions to a fixed size

Convert images into feature vectors

Train a neural network

Evaluate performance on training, validation, and test sets

The implementation uses only:

C++

OpenCV Machine Learning module

No external deep learning frameworks are used.

Dataset Structure

The dataset follows a YOLO-style directory structure.

data/

train/
    images/
        img1.jpg
        img2.jpg
    labels/
        img1.txt
        img2.txt

val/
    images/
    labels/

test/
    images/
    labels/

Each image has a corresponding label file with the same name.

Example:

img1.jpg
img1.txt
YOLO Label Format

Each label file contains bounding box annotations in YOLO format.

<class_id> <x_center> <y_center> <width> <height>

Example:

0 0.45 0.51 0.30 0.26
1 0.60 0.40 0.22 0.19

Class mapping:

0 = Smoke
1 = Fire

All coordinates are normalized between 0 and 1.

Image Processing Pipeline

For each image, the following steps are performed:

Read the image from disk

Load the corresponding label file

Convert YOLO coordinates into an OpenCV rectangle

Crop the region of interest from the image

Resize the region to 32 × 32 pixels

Normalize pixel values

Flatten the image into a feature vector

Final feature size:

32 × 32 × 3 = 3072
Program Flow
Start
   |
Load Training Dataset
   |
Extract Bounding Boxes
   |
Crop Image Regions
   |
Resize (32x32)
   |
Normalize Pixels
   |
Flatten → Feature Vector
   |
Train ANN_MLP Model
   |
Evaluate (Train / Val / Test)
   |
Save Model (.xml)
   |
End
Neural Network Architecture

The model is a Multilayer Perceptron (MLP).

Input Layer     : 3072
Hidden Layer 1  : 512
Hidden Layer 2  : 128
Hidden Layer 3  : 32
Output Layer    : 2

Activation Function:

SIGMOID_SYM

Training Method:

Backpropagation

Training Iterations:

100
Requirements

The project requires:

C++17

OpenCV 4.x

A compiler supporting std::filesystem

Build Instructions

Compile using g++:

g++ main.cpp -o smoke_fire_nn `pkg-config --cflags --libs opencv4` -std=c++17
Running the Program

Run the compiled executable:

./smoke_fire_nn
Example Console Output
Loading training set
Loaded 500 images
Loaded 1000 images
Loaded 1500 images

Training neural network...
Training finished

Train Accuracy: 94.8%
Validation Accuracy: 91.6%
Test Accuracy: 90.9%
Output Model

After training, the neural network is saved as:

smoke_fire_nn.xml

The model can be loaded later using:

Ptr<ANN_MLP> model = ANN_MLP::load("smoke_fire_nn.xml");
