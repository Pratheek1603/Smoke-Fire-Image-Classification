# Smoke-Fire-Image-Classification
A C++ machine learning project that trains a neural network using OpenCV's ANN_MLP to classify Smoke vs Fire from images using bounding box annotations in YOLO format.
Smoke / Fire Classification using OpenCV Neural Network (C++)

This project trains a simple neural network in C++ using OpenCV's ANN_MLP module to classify smoke and fire from image regions.
The training data consists of 14,000+ images with YOLO format bounding box annotations.

The program extracts object regions from the images using the YOLO labels, preprocesses them, and trains a fully connected neural network to perform binary classification.

The final trained model is saved as:

smoke_fire_nn.xml

which can later be loaded for inference.

Project Overview

The main idea is to:

Load images and YOLO annotations

Extract the labeled regions

Resize them to a fixed size

Convert them into feature vectors

Train a neural network

Evaluate accuracy on train / validation / test sets

This implementation uses only OpenCV + C++, without external deep learning frameworks.

Dataset Format

The dataset is expected in a structure similar to common YOLO datasets.

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

Each label file contains bounding boxes in YOLO format:

<class_id> <x_center> <y_center> <width> <height>

Example:

0 0.45 0.51 0.30 0.26
1 0.60 0.40 0.22 0.19

Where:

0 = Smoke
1 = Fire

Coordinates are normalized between 0 and 1.

Processing Pipeline

For every image:

Read the image

Load corresponding label file

Convert YOLO coordinates to OpenCV rectangle

Crop the region from the image

Resize to 32x32

Normalize pixel values

Flatten into a feature vector

The final feature size:

32 x 32 x 3 = 3072
Program Flow
                Start
                  |
                  v
        Load Training Dataset
                  |
                  v
        Extract Bounding Boxes
                  |
                  v
        Crop Image Regions
                  |
                  v
          Resize (32x32)
                  |
                  v
         Normalize Pixels
                  |
                  v
      Flatten → Feature Vector
                  |
                  v
         Train ANN_MLP Model
                  |
                  v
      Evaluate (Train / Val / Test)
                  |
                  v
        Save Model (.xml)
                  |
                  v
                 End
Neural Network Architecture

The network is a fully connected multilayer perceptron.

Input Layer     : 3072
Hidden Layer 1  : 512
Hidden Layer 2  : 128
Hidden Layer 3  : 32
Output Layer    : 2

Activation function:

SIGMOID_SYM

Training method:

Backpropagation

Training iterations:

100
Building the Project

Requirements:

C++17

OpenCV 4.x

Compiler with filesystem support

Compile with g++:

g++ main.cpp -o smoke_fire_nn `pkg-config --cflags --libs opencv4` -std=c++17

Run the program:

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

After training finishes the model is stored as:

smoke_fire_nn.xml

The model can be loaded later using:

Ptr<ANN_MLP> model = ANN_MLP::load("smoke_fire_nn.xml");
Notes

Regions smaller than 20x20 pixels are ignored.

Each cropped region is resized to 32×32 before training.

Labels are converted to one-hot encoded vectors for training.
