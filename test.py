import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("xoblank.keras")

classNames = ['o', 'x', 'blank']

img = Image.open("test_x.jpeg").convert("L")
img = img.resize((64, 64))

imgArr = np.array(img)
imgArr = np.expand_dims(imgArr, axis=-1)
imgArr = np.expand_dims(imgArr, axis=0)

prediction = model.predict(imgArr)

predictedLet = np.argmax(prediction[0])

print("Prediction:", class_names[predictedLet])
print("Confidence:", np.max(prediction[0]))
print(classNames)
print(prediction[0])
