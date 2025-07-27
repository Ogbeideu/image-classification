# Cat vs Dog CNN Classifier

Convolutional Neural Network for binary image classification using Keras. Classifies images as cats (0) or dogs (1) with data augmentation.

## Model Architecture
- **3-layer CNN**: Conv2D → MaxPool → Flatten → Dense
- **Input**: 150x150x3 RGB images  
- **Parameters**: 1.2M trainable parameters
- **Accuracy**: ~55% (small dataset limitation)

## Tech Stack
- **Keras/TensorFlow** - CNN model
- **ImageDataGenerator** - Data augmentation
- **Google Colab** - Training environment

## Quick Start
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), input_shape=(150,150,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Dataset
- **198 training images** (cats & dogs)
- **100 validation images**
- **Data augmentation**: rescaling, shear, zoom, flip

## Key Features
- Binary classification with sigmoid activation
- Data augmentation to prevent overfitting
- Real-time prediction visualization
- Training history plots

## Usage
1. Mount Google Drive with image dataset
2. Run CNN training notebook
3. Evaluate model performance
4. Test predictions on new images

## Installation
```bash
pip install tensorflow keras matplotlib numpy
```

*Deep Learning • Computer Vision • CNN*
