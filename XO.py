import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = 64
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)
print(train_ds.class_names)
# val_ds = tf.keras.utils.image_dataset_from_directory(
#     "dataset/val",
#     image_size=(IMG_SIZE, IMG_SIZE),
#     color_mode="grayscale",
#     batch_size=BATCH_SIZE,
#     label_mode="int"
# )


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(64, 64, 1)),
#     tf.keras.layers.Rescaling(1./255),

#     tf.keras.layers.RandomRotation(0.2),
#     tf.keras.layers.RandomTranslation(0.1, 0.1),
#     tf.keras.layers.RandomZoom(0.1),
#     tf.keras.layers.RandomContrast(0.3),

#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(3, activation='softmax')
# ])

# model.compile(optimizer = tf.keras.optimizers.Adam(),
#               loss = 'sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=20
# )
# model.save("xoblank.keras")


