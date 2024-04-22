import keras.models
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

model = load_model('models/fbsi.h5')
target_size = (224, 224)

img_path = 'images/test1.jpg'
img = load_img(img_path, target_size=target_size)

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
if img_array.shape[-1] != 3:
    raise ValueError("La imagen no tiene 3 canales de color (RGB)")

img_array = img_array / 224.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

print('Prediccion => ', predicted_class)