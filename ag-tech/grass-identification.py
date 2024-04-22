from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

labels = np.genfromtxt('dataset/fbsi/annotations.csv', delimiter=',', dtype=str, skip_header=1)

image_paths = ['dataset/' + img_name + '.jpg' for img_name in labels[:, 0]]

images = [load_img(img_path, target_size=(224, 224)) for img_path in image_paths]
images = [img_to_array(img) for img in images]
images = np.array(images)

images = images / 255.0

labels_dict = {'0': 0, '0.5':0.5, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
labels = np.array([labels_dict[label] for label in labels[:, 6]])

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.save('models/fbsi.h5')

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)