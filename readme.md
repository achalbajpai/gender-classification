# Gender Classification using Deep Learning

A deep learning model for gender classification using facial images, built with PyTorch and Streamlit.

## Features

-  Deep learning model based on ResNet18 architecture
-  Transfer learning with pre-trained weights
-  Data augmentation pipeline
-  Real-time training visualization
-  Interactive web interface using Streamlit
-  Comprehensive performance metrics

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/yourusername/gender-classification.git
cd gender-classification
```

2. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare dataset:

-  Place your dataset in the `Dataset` folder with the following structure:

```
Dataset/
├── Train/
│   ├── Female/
│   └── Male/
├── Validation/
│   ├── Female/
│   └── Male/
└── Test/
    ├── Female/
    └── Male/
```

5. Train the model:

```bash
python train.py
```

6. Run the web interface:

```bash
streamlit run app.py
```

## Model Architecture

-  Base Model: ResNet18 (pre-trained on ImageNet)
-  Modified for binary classification
-  Input size: 128x128 RGB images
-  Data augmentation: horizontal flips, rotations, color jittering

## Training Details

-  Optimizer: Adam
-  Learning rate: 0.001 with reduction on plateau
-  Loss function: Binary Cross-Entropy with Logits
-  Early stopping with patience=5
-  Batch size: 64
-  Training samples: 20,000
-  Validation samples: 5,000
-  Test samples: 5,000

## Performance

-  Target accuracy: 90%+
-  Training time: ~1 hour on CPU
-  Metrics tracked:
   -  Accuracy
   -  Loss
   -  ROC curve
   -  Confusion matrix

## Web Interface

The Streamlit interface provides:

1. Model testing with image upload
2. Real-time predictions
3. Confidence visualization
4. Performance metrics display
5. Technical documentation

## Files Description

-  `train.py`: Model training script
-  `app.py`: Streamlit web interface
-  `requirements.txt`: Project dependencies
-  `technical_details.md`: Detailed technical documentation
-  `learnthis.md`: Learning resources and explanations

## Requirements

-  Python 3.9+
-  PyTorch 1.9+
-  CUDA (optional, for GPU acceleration)
-  Other dependencies in requirements.txt

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
