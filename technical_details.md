# Gender Classification Model Technical Details

## Model Architecture

-  Base Model: ResNet18 (pre-trained on ImageNet)
-  Modified for binary classification
-  Input size: 128x128x3 (RGB images)
-  Final layer: Fully connected with dropout (0.3)

## Model Parameters

### Trainable Parameters

-  All layers except the base layers are trainable
-  Final fully connected layer: in_features → 1 (binary classification)
-  Total parameters: ~11.7 million

### Training Configuration

-  Loss Function: Binary Cross Entropy with Logits
-  Optimizer: Adam (Learning rate: 0.001)
-  Learning Rate Scheduler: ReduceLROnPlateau
   -  Mode: min
   -  Factor: 0.2
   -  Patience: 3
-  Early Stopping:
   -  Patience: 5
   -  Monitor: Validation Loss

## Dataset Information

### Structure

-  Location: /Users/achal/Downloads/gender-classification/Dataset
-  Split into:
   -  Training set: 20,000 images
   -  Validation set: 5,000 images
   -  Test set: 5,000 images

### Class Distribution in Test Set

-  Female: 2,851 images
-  Male: 2,149 images

### Data Preprocessing

-  Resize: 128x128
-  Data Augmentation (Training only):
   -  Random Horizontal Flip
   -  Random Rotation (±10 degrees)
   -  Color Jitter (brightness, contrast)
-  Normalization:
   -  Mean: [0.485, 0.456, 0.406]
   -  Std: [0.229, 0.224, 0.225]

### Data Loading

-  Batch Size: 64
-  Shuffle: True (training) / False (validation, test)
-  Number of workers: 4

## Performance Metrics

### Final Results

-  Test Accuracy: 98.22%
-  Precision:
   -  Female: 0.98
   -  Male: 0.98
-  Recall:
   -  Female: 0.99
   -  Male: 0.98
-  F1-Score:
   -  Female: 0.98
   -  Male: 0.98

## Training Process

-  Maximum Epochs: 20
-  Early Stopping enabled
-  Learning rate reduction on plateau
-  Training time: ~1 hour on CPU
-  Visualizations:
   -  Training curves saved as 'training_metrics.png'
   -  Confusion matrix saved as 'confusion_matrix.png'
   -  ROC curve saved as 'roc_curve.png'

## Model Output

-  Binary classification (Female/Male)
-  Output activation: Sigmoid
-  Decision threshold: 0.5
-  Confidence score provided for predictions
