# Smoke / Fire Classification — OpenCV Neural Network (C++)

This project trains a neural network in C++ using OpenCV's `ANN_MLP` module to classify smoke and fire from image regions. No external deep learning frameworks — just C++ and OpenCV.

The dataset has 14,000+ images with YOLO-format bounding box annotations. The program extracts object regions, preprocesses them into feature vectors, and trains a fully connected neural network for binary classification. The trained model is saved as `smoke_fire_nn.xml` and can be loaded later for inference.

## Results

| Dataset    | Accuracy |
|------------|----------|
| Train      | 99.82%   |
| Validation | 91.64%   |
| Test       | 90.95%   |

## What it does

1. Loads images and YOLO annotations
2. Crops the labeled bounding box regions
3. Resizes each region to 32×32
4. Normalizes and flattens into a 3072-value feature vector
5. Trains a fully connected MLP
6. Evaluates on train / val / test sets
7. Saves the model to `smoke_fire_nn.xml`

## Network Architecture

```
Input Layer      →  3072  (32 × 32 × 3)
Hidden Layer 1   →  512
Hidden Layer 2   →  128
Hidden Layer 3   →  32
Output Layer     →  2
```

Activation: `SIGMOID_SYM`, trained with backpropagation for 100 iterations.

## Dataset

Uses the [Smoke / Fire Detection YOLO dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) from Kaggle.

Download it and extract into the project so the structure looks like this:

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each label file matches its image by name (e.g. `img1.jpg` → `img1.txt`) and follows YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Class mapping: `0 = Smoke`, `1 = Fire`. All coordinates normalized 0–1.

## Requirements

- C++17
- OpenCV 4.x
- Compiler with `std::filesystem` support

## Build

```bash
g++ main.cpp -o smoke_fire_nn `pkg-config --cflags --libs opencv4` -std=c++17
```

## Run

```bash
./smoke_fire_nn
```

Output will look something like:

```
Loading training set...
Loaded 500 images
Loaded 1000 images
...
Training neural network...
Training finished

Train Accuracy:      99.8161%
Validation Accuracy: 91.6376%
Test Accuracy:       90.9496%
```

## Loading the model for inference

```cpp
Ptr<ANN_MLP> model = ANN_MLP::load("smoke_fire_nn.xml");
```
