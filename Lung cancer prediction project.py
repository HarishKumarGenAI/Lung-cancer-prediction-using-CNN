import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Rescaling
)
from tensorflow.keras.optimizers import Adam

# =====================================================
# PROJECT PATHS 
# =====================================================

BASE_DIR = r"E:\Lung cancer_prediction"

DATASET_DIR = os.path.join(BASE_DIR, "lung_subset_small_folder")

# =====================================================
# SETTINGS
# =====================================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# =====================================================
# LOAD TRAIN DATASET
# =====================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# =====================================================
# LOAD VALIDATION DATASET
# =====================================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# =====================================================
# CLASS NAMES
# =====================================================

class_names = train_ds.class_names

print("Classes:", class_names)

os.makedirs(BASE_DIR, exist_ok=True)

with open(
    os.path.join(BASE_DIR, "class_indices.json"),
    "w"
) as f:
    json.dump(class_names, f)

# =====================================================
# NORMALIZATION
# =====================================================

normalization_layer = Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)

# =====================================================
# CNN MODEL
# =====================================================

model = Sequential([

    Input(shape=(128, 128, 3)),

    Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        3,
        activation='softmax'
    )
])

# =====================================================
# OPTIMIZER WITH LEARNING RATE
# =====================================================

optimizer = Adam(
    learning_rate=LEARNING_RATE
)

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# MODEL SUMMARY
# =====================================================

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =====================================================
# SAVE MODEL
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "lung_cancer_model.keras"
)

model.save(MODEL_PATH)

print("Model Saved:", MODEL_PATH)

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history['accuracy'],
    label="Training Accuracy"
)

plt.plot(
    history.history['val_accuracy'],
    label="Validation Accuracy"
)

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()