import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Modell laden
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Input/Output Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels
class_names = open("labels.txt", "r").readlines()

# Bild laden
image = Image.open("Herunterladen (1).jpg").convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Bild in Input laden
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(normalized_image_array, axis=0).astype(np.float32))

# Inferenz
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

index = np.argmax(output_data)
class_name = class_names[index]
confidence_score = output_data[0][index]

print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)