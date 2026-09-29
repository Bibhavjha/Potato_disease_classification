# Technical Documentation — Potato Disease Classification

## 1. Problem Statement

Early and late blight are among the most damaging fungal diseases affecting potato crops worldwide. Manual identification requires expert knowledge and is time-consuming, especially for smallholder farmers. This project explores whether a convolutional neural network (CNN) can reliably distinguish between healthy leaves and the two blight types from a single leaf image, as a foundation for future low-cost, automated crop-monitoring tools.

## 2. Dataset

- **Name**: PlantVillage (potato subset)
- **Classes** (3): Healthy, Early Blight, Late Blight
- **Total samples**: 2,152 images
- **Format**: Directory structure with one subfolder per class, loaded via `tf.keras.preprocessing.image_dataset_from_directory`
- **Image size**: All images resized to 256 × 256 pixels, 3 color channels (RGB)
- **Batch size**: 32

### 2.1 Data Splitting

The dataset was split using a custom function (`get_dataset_partitions_tf`) rather than a fixed slice, to allow reproducible, shuffled splitting:

- Training: 80%
- Validation: 10%
- Test: 10%
- Shuffle seed: 12 (fixed for reproducibility)
- Shuffle buffer size: 1,000

### 2.2 Performance Optimization

Each split (`train_dataset`, `val_dataset`, `test_dataset`) was optimized using:

- `.cache()` — keeps data in memory after first epoch to avoid repeated disk reads
- `.shuffle(1000)` — randomizes batch order
- `.prefetch(buffer_size=tf.data.AUTOTUNE)` — overlaps data preprocessing with model execution

## 3. Preprocessing & Augmentation

Two preprocessing stages were built as Keras `Sequential` layers, applied inline within the model so they run identically at both training and inference time:

**Resize and Rescale** (applied to all data):
- `Resizing(256, 256)`
- `Rescaling(1.0 / 255)` — normalizes pixel values to the [0, 1] range

**Data Augmentation** (applied only during training, via the model pipeline):
- `RandomFlip("horizontal_and_vertical")`
- `RandomRotation(0.2)` — random rotation up to ±20%

Augmentation was included specifically to reduce overfitting given the moderate dataset size (2,152 images across 3 classes), by artificially increasing the diversity of leaf orientations the model sees during training.

## 4. Model Architecture

A CNN was built from scratch (no pretrained backbone / transfer learning). The architecture consists of 4 convolution + max-pooling blocks followed by a small dense classifier head:

| Layer | Output Shape / Config |
|---|---|
| Input | 256 × 256 × 3 |
| Resizing + Rescaling | 256 × 256 × 3 |
| Data Augmentation | 256 × 256 × 3 |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 64 units, ReLU |
| Dense (output) | 3 units, Softmax |

**Design rationale**: filter count doubles after the first block (32 → 64) and then holds steady, a common pattern that lets the network build progressively more abstract features while controlling parameter growth. Four pooling stages reduce the 256×256 input down to a compact spatial resolution before flattening, keeping the dense layers small and reducing overfitting risk on a dataset of this size.

## 5. Training Configuration

- **Optimizer**: Adam (default learning rate)
- **Loss function**: Sparse Categorical Crossentropy (`from_logits=False`, since the output layer uses softmax)
- **Metric**: Accuracy
- **Epochs**: 15
- **Batch size**: 32

## 6. Results

| Metric | Value |
|---|---|
| Test accuracy | 91.4% |
| Test loss | 0.207 |

Training and validation accuracy/loss were tracked per epoch and plotted to visually inspect for overfitting (training vs. validation curves). Qualitative predictions were also inspected by sampling individual test images, displaying the true label, predicted label, and the model's confidence score for each.

## 7. Evaluation Methodology

- **Quantitative**: overall test-set accuracy and loss via `model.evaluate()`.
- **Qualitative**: a sample of 9 test images was visualized alongside their actual label, predicted label, and prediction confidence, to sanity-check that predictions were reasonable beyond the aggregate accuracy number.

## 8. Limitations

- **No confusion matrix**: the current evaluation does not break down which specific classes are most often confused (e.g., whether Early Blight and Late Blight — which can look visually similar — are the primary source of the ~8.6% test error).
- **No comparison against a transfer-learning baseline** (e.g., fine-tuned ResNet18/MobileNetV2), which would indicate whether the from-scratch CNN is competitive with standard approaches.
- **No interpretability analysis** (e.g., Grad-CAM) to confirm the model is attending to disease-relevant regions of the leaf rather than incidental background features.
- **Single dataset, single domain**: PlantVillage images are taken under relatively controlled conditions; performance on field-collected photos with variable lighting, angles, and backgrounds is untested.
- **No deployment/inference packaging** (e.g., TensorFlow Lite conversion) is included in this version of the project.

## 9. Possible Future Extensions

- Add a confusion matrix and per-class precision/recall/F1 scores.
- Compare against a transfer-learning model (ResNet18, MobileNetV2) pretrained on ImageNet.
- Apply Grad-CAM or similar techniques to visualize model attention.
- Test on external, field-collected images to assess real-world generalization.
- Package the trained model (e.g., TensorFlow Lite) for lightweight mobile/edge deployment.

## 10. Environment

- **Framework**: TensorFlow / Keras
- **Language**: Python
- **Supporting libraries**: Matplotlib (visualization), NumPy

## Author

Bibhav Jha
