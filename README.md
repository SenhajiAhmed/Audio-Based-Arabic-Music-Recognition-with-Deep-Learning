# Arabic Music Genre Classification

A deep learning model for classifying Arabic music genres using Mel-frequency Cepstral Coefficients (MFCC) features and Convolutional Neural Networks (CNN).

## Overview

This project implements a CNN-based classifier that can identify different Arabic music genres from audio files. The model processes audio clips by extracting MFCC features and uses a multi-layer convolutional architecture for classification.

## Features

- **Audio Feature Extraction**: Uses MFCC coefficients for robust audio representation
- **CNN Architecture**: Multi-layer convolutional neural network with batch normalization and dropout
- **Flexible Audio Processing**: Can handle audio files of any length by processing them in chunks
- **Comprehensive Evaluation**: Includes confusion matrix, classification reports, and training curves
- **Real-time Prediction**: Process new audio files and get genre predictions with confidence scores

## Dataset Requirements

The model expects a JSON file containing preprocessed MFCC features.

## Audio Configuration

- **Sampling Rate**: 22,050 Hz
- **Training Duration**: 10 seconds per clip
- **Prediction Duration**: 30 seconds per chunk
- **MFCC Features**: 13 coefficients
- **Time Steps**: 130 (for training), dynamically handled for prediction

## Model Architecture

```
Conv2D(32) → BatchNorm → MaxPooling → Dropout(0.3)
Conv2D(64) → BatchNorm → MaxPooling → Dropout(0.3) 
Conv2D(128) → BatchNorm → MaxPooling → Dropout(0.4)
Flatten → Dense(256) → Dropout(0.5) → Dense(num_genres)
```

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. If using GPU (recommended):
```bash
pip install tensorflow-gpu
```

## Usage

### Training the Model

1. Prepare your dataset in JSON format
2. Update the `JSON_PATH` variable with your dataset path
3. Run the training script:

```python
# The training will automatically:
# - Load and preprocess the data
# - Split into train/validation/test sets
# - Train the CNN model
# - Save the best model based on validation accuracy
```

### Making Predictions

```python
import tensorflow as tf
import librosa
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("arabic_music_genre_model.keras")

# Process an audio file
result = process_any_audio("path/to/your/audio.wav", chunk_size=30)

# Access results
print(f"Predicted genre: {result['predicted_genre']}")
print(f"Confidence scores: {result['average_confidences']}")
```

### Example Output

```
===== RESULTS =====
Predicted genre: Classical
Processed 4 chunks of 30 seconds each

Genre distribution across chunks:
  Classical: 3 chunks (75.0%)
  Folk: 1 chunks (25.0%)

Average confidence scores:
  Classical: 87.45%
  Folk: 65.23%
  Pop: 45.12%
```

## Model Performance

The model includes several techniques to improve performance:

- **Class Weight Balancing**: Automatically handles imbalanced datasets
- **Early Stopping**: Prevents overfitting with patience of 10 epochs
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Layers**: Reduces overfitting with varying dropout rates
- **Data Augmentation**: Through TensorFlow's data pipeline

## File Structure

```
arabic-music-recognition/
│
├── README.md
├── requirements.txt
├── Arabic_Music_Recognition.ipynb    # Main notebook
├── arabic_music_genre_model.keras   # Trained model
└── data/
    └── Finaldata.json              # Preprocessed dataset
```

## Key Functions

### `load_dataset(json_path)`
Loads preprocessed MFCC features and labels from JSON file.

### `create_dataset(X, y, batch_size, shuffle)`
Creates optimized TensorFlow datasets with batching and prefetching.

### `extract_features(audio_signal)`
Extracts MFCC features from raw audio signal.

### `process_any_audio(file_path, chunk_size)`
Processes audio files of any length by chunking and provides detailed predictions.

## Configuration Parameters

You can modify these parameters based on your needs:

```python
SR = 22050          # Sampling rate
DURATION = 10       # Training clip duration (seconds)
N_MFCC = 13        # Number of MFCC coefficients
N_FFT = 2048       # FFT window size
HOP_LENGTH = 1723  # Hop length for training
BATCH_SIZE = 64    # Training batch size
```

## GPU Support

The code automatically detects and configures GPU usage if available:
```python
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## Evaluation Metrics

The model provides comprehensive evaluation including:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Visual representation of predictions vs actual labels
- **Classification Report**: Precision, recall, and F1-score for each genre
- **Training Curves**: Loss and accuracy plots over epochs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `BATCH_SIZE` if you encounter OOM errors
2. **Audio Loading**: Ensure librosa can read your audio format
3. **Model Loading**: Check file paths and ensure the model file exists

### Audio Format Support

The model supports any audio format that librosa can handle:
- WAV, MP3, FLAC, M4A, etc.
- Automatically resampled to 22,050 Hz

## Future Improvements

- Add data augmentation techniques (pitch shifting, time stretching)
- Implement ensemble methods for better accuracy
- Add real-time audio stream processing
- Support for longer audio contexts
- Integration with music streaming APIs