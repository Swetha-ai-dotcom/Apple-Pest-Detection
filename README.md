# Apple Disease Detection

## Overview
This project aims to classify apple diseases based on images. The model is trained to detect various apple diseases, such as rot, scab, and blotch, and also identifies healthy apples. The goal is to assist in apple orchard management by automating the identification of diseases, allowing for faster intervention and better crop management.

---

## Table of Contents

1. [Project Setup](#project-setup)
2. [Data Description](#data-description)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Model Evaluation](#model-evaluation)
6. [Prediction](#prediction)
7. [Usage](#usage)
8. [Requirements](#requirements)

---

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/apple-disease-detection.git
cd apple-disease-detection
```

### 2. Install dependencies

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

---

## Data Description

The dataset consists of images of apples categorized into the following classes:

- **Rot Apple**: Apples affected by rot, which is classified under Grading C.
- **Scab Apple**: Apples with scab disease, classified under Grading D.
- **Blotch Apple**: Apples with blotch disease, classified under Grading B.
- **Healthy Apple**: Apples without any disease, classified under Grading A.

### Dataset Structure

The dataset is divided into two parts:

1. **Training Data**: 
   - Stored in the `data/Train` directory.
   - Includes images for each disease category.
   
2. **Testing Data**: 
   - Stored in the `data/Test` directory.
   - Includes images for validation and testing.

---

## Model Architecture

The model is based on a Convolutional Neural Network (CNN), which is commonly used for image classification tasks. The architecture is defined as follows:

1. **Input Layer**: Accepts images of size (224, 224, 3).
2. **Convolutional Layers**: Four convolutional layers followed by max-pooling layers to extract features.
3. **Dropout Layer**: To prevent overfitting.
4. **Dense Layer**: A fully connected layer with 512 neurons.
5. **Output Layer**: A softmax layer with 4 neurons corresponding to the 4 categories (Rot, Scab, Blotch, and Healthy).

### Model Summary

```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 222, 222, 128)     3584      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 111, 111, 128)     0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 109, 109, 128)     147584    
_________________________________________________________________
max_pooling2d_1 (MaxPooling (None, 54, 54, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 52, 52, 64)        73792     
_________________________________________________________________
max_pooling2d_2 (MaxPooling (None, 26, 26, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 24, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling (None, 12, 12, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dropout (Dropout)            (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               4719104   
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 2052      
=================================================================
Total params: 4,891,044
Trainable params: 4,891,044
Non-trainable params: 0
_________________________________________________________________
```

---

## Training

To train the model, we used the following settings:

- **Image Size**: 224x224 pixels.
- **Batch Size**: 64.
- **Epochs**: 150.
- **Optimizer**: RMSprop.
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy.

### Data Augmentation

Data augmentation techniques were used to improve the model's generalization, such as:

- Rotation
- Width and height shift
- Shear and zoom transformations
- Horizontal flipping

---

## Model Evaluation

Once the model is trained, you can evaluate its performance using the validation set. The training history includes information about the loss and accuracy, which you can plot to visualize the model's performance during training.

### Training and Validation Curves

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()
```

---

## Prediction

After training, you can use the model to make predictions on new apple images. Here is a code example to predict an image:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = load_model('final_model.keras')

# Path to the image
img_path = 'path_to_image.jpg'

# Load the image
img = load_img(img_path, target_size=(224, 224))

# Convert image to array and expand dimensions
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print(f"Predicted class: {predicted_class}")
```

---

## Usage

1. Clone the repository and install dependencies.
2. Place your apple images in the appropriate directories (`Train` or `Test`).
3. Run the training script to train the model.
4. Use the trained model to make predictions on new apple images.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
