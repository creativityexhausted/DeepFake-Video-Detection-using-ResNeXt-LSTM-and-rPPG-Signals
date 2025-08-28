# DeepFake Video Detection using ResNeXt-LSTM and rPPG Signals

A deep learning framework for detecting deepfake videos by combining advanced spatial, temporal, and physiological analysis. This hybrid approach integrates a ResNeXt50 convolutional neural network for spatial feature extraction, an LSTM for sequence modeling, and rPPG (remote Photoplethysmography)-derived physiological signals, resulting in improved classification accuracy for deepfake detection.

## Features

- **Hybrid Architecture**: Merges ResNeXt50 CNN for spatial representation with sequential modeling using LSTM.
- **Physiologically-Informed Classification**: Augments neural predictions with rPPG-based physiological cues extracted from facial regions.
- **End-to-End Pipeline**: Automated workflows from pre-processing, feature extraction, model training, and evaluation.
- **Modular Codebase**: Extensible modules for each stage—customize components with ease.
- **Comprehensive Documentation**: Well-annotated code for reproducibility and clarity.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)


## Usage

#### Detect DeepFakes from Video

```python
from deepfake_detector import DeepFakeDetector

detector = DeepFakeDetector('resnext50_lstm_rppg.pth')
result = detector.predict('sample_video.mp4')
print(f'Prediction: {result}')
```

#### Command-Line Interface

```bash
python run_detector.py --video sample_video.mp4 --output results.json
```

## Model Overview

| Component     | Purpose                                |
|---------------|----------------------------------------|
| ResNeXt50     | Spatial feature extraction from frames |
| LSTM          | Temporal sequence modeling             |
| rPPG Signals  | Capture physiological facial cues      |
| Classifier    | Combines all features for prediction   |

## Data Preparation

- Organize video datasets as follows:
  ```
  dataset/
    real/
      video1.mp4
      video2.mp4
    fake/
      video3.mp4
      video4.mp4
  ```
- Use the provided scripts in `data_preprocessing/` to:
  - Extract frames
  - Detect and align faces
  - Compute rPPG signals from video

## Training

```bash
python train.py --config configs/default.yaml
```

Options include hyperparameter tuning, dataset selection, and model architecture customization.

## Evaluation

- Evaluate classification accuracy, AUC, precision, and recall on the test set:
  
  ```bash
  python evaluate.py --model checkpoints/best_model.pth --data test_dataset/
  ```
- Generate confusion matrix and ROC curves for detailed analysis.

## Results

- Significant accuracy improvement over pure CNN or RNN models on standard deepfake benchmarks.
- Improved detection, especially for videos with subtle manipulations, by incorporating physiological (rPPG) features.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for any bugs, feature requests, or improvements.
