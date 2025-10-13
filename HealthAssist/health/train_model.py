import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers

# --- Paths ---
train_dir = "dataset/train"
val_dir = "dataset/val"

# --- Ensure paths exist ---
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("Dataset directories not found. Please check 'dataset/train' and 'dataset/val'.")

# --- Data Preprocessing & Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# --- Load Pretrained MobileNetV2 ---
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers for transfer learning
base_model.trainable = False

# --- Build Model ---
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

# --- Compile Model ---
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Train Model ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)


# --- Save Model ---
model.save("skin_disease_model.h5")
print("✅ Model saved as 'skin_disease_model.h5'")

# --- Save Class Labels ---
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=4)
print("✅ Class indices saved as 'class_indices.json'")
