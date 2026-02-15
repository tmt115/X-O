import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 64

model = tf.keras.models.load_model("xoblank.keras")

class_names = ['o', 'x', 'blank']

img = Image.open("test_x.jpeg").convert("L")
img = img.resize((IMG_SIZE, IMG_SIZE))

img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=-1)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions[0])

print("Prediction:", class_names[predicted_class])
print("Confidence:", np.max(predictions[0]))
print(class_names)
print(predictions[0])