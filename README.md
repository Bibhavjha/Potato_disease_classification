# Potato Disease Classification using CNN

A convolutional neural network that classifies potato leaf images into three categories — **Healthy**, **Early Blight**, and **Late Blight** — built with TensorFlow/Keras and trained on the PlantVillage dataset.

## Overview

Potato crops are commonly affected by early blight and late blight, two fungal diseases that can significantly reduce yield if not identified early. This project builds an image classification pipeline that takes a photo of a potato leaf and predicts whether it is healthy, affected by early blight, or affected by late blight — a step toward automated, low-cost crop health monitoring.

## Dataset

- **Source**: [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (potato subset)
- **Classes**: 3 — `Potato___healthy`, `Potato___Early_blight`, `Potato___Late_blight`
- **Total images**: 2,152
- **Image size**: resized to 256 × 256 × 3 (RGB)

## Results

| Metric | Value |
|---|---|
| Test accuracy | **91.4%** |
| Test loss | 0.207 |
| Epochs trained | 15 |
| Batch size | 32 |

## Model Architecture

A CNN built from scratch (no transfer learning) with 4 convolutional blocks:

```
Input (256x256x3)
  → Resizing + Rescaling (1/255)
  → Data Augmentation (random flip, random rotation)
  → Conv2D(32, 3x3, relu) → MaxPooling2D
  → Conv2D(64, 3x3, relu) → MaxPooling2D
  → Conv2D(64, 3x3, relu) → MaxPooling2D
  → Conv2D(64, 3x3, relu) → MaxPooling2D
  → Flatten
  → Dense(64, relu)
  → Dense(3, softmax)
```

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metric**: Accuracy

## Data Pipeline

- Loaded using `tf.keras.preprocessing.image_dataset_from_directory`
- Split 80% train / 10% validation / 10% test using a custom partitioning function with shuffling (seed=12)
- Performance optimizations: `.cache()`, `.shuffle()`, and `.prefetch(AUTOTUNE)` applied to all splits
- Augmentation applied only during training: random horizontal/vertical flip and random rotation (±20%), to improve generalization on a relatively small dataset

## How to Run

1. Install dependencies:
   ```
   pip install tensorflow matplotlib numpy
   ```
2. Download the PlantVillage potato subset and place it in a folder named `PlantVillage/` with one subfolder per class.
3. Open and run `training.ipynb` top to bottom in Jupyter or Google Colab (GPU recommended).

## Project Structure

```
├── training.ipynb        # Data loading, model, training, evaluation
├── PlantVillage/          # Dataset (not included, download separately)
└── README.md
```

## Limitations & Future Work

- Trained on a single, relatively small dataset (2,152 images); performance on real-world field photos (varying lighting, backgrounds, camera quality) is untested.
- No class-wise error analysis yet — a confusion matrix would clarify whether misclassifications mostly occur between the two visually similar blight types.
- Model was trained from scratch; a transfer-learning approach (e.g., fine-tuning a pretrained ResNet or MobileNet) could be compared as a stronger baseline.
- No model interpretability step (e.g., Grad-CAM) yet, to visualize which image regions drive each prediction.

## Author

Bibhav Jha
