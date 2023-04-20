#############
# Classify pictures from multiple Logos
#############


# Setup
import tensorflow as tf
from tensorflow.keras.layers import RNN
import os

from sklearn.preprocessing import LabelEncoder

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Remove dodgy images
import cv2
import imghdr

data_dir = "/Users/paulinaheine/Codes/Vodafone_Inventur/Logos"
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

ds_store_file_location = '/Users/paulinaheine/Codes/Vodafone_Inventur/Logos/.DS_store'
if os.path.isfile(ds_store_file_location):
    os.remove(ds_store_file_location)

''' Braucht ,man anscheinend nicht
for image_class in os.listdir("/Users/paulinaheine/Codes/Vodafone_Inventur/Logos"):
    os.listdir(os.path.join("/Users/paulinaheine/Codes/Vodafone_Inventur/Logos", image_class))
    ds_store_file_location = os.listdir(os.path.join("/Users/paulinaheine/Codes/Vodafone_Inventur/Logos", image_class,".DS_store"))
    if os.path.isfile(ds_store_file_location):
        os.remove(ds_store_file_location)
'''

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)

# Load data
import numpy as np
from matplotlib import pyplot as plt

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

'''
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
'''

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

'''
model = Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
'''

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical",
                                   input_shape=(img_height,
                                                img_width,
                                                3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomCrop(height=img_height,
                                   width=img_width),
        tf.keras.layers.RandomContrast(0.5, 1.5)
    ]
)
'''
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
'''

model = Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 25

# Wirkt Overfitting entgegen
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=5, verbose=1)
# callback = [early_stop]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    # callbacks = callback
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

'''
#Hua, cisco, sun,sun
test = ["/Users/paulinaheine/Codes/Vodafone_Inventur/Logos/Huawei/images31.jpg","/Users/paulinaheine/IMG_3083.png","/Users/paulinaheine/IMG_3120.png","/Users/paulinaheine/IMG_3121.png"]

for i in range(len(test)):
    img = tf.keras.utils.load_img(
        test[i], target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    for i in range(5):
        print(score[i])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


## test all
scores = []
for image_class in os.listdir("/Users/paulinaheine/Codes/Vodafone_Inventur/Logos"):
    full = len(os.listdir(os.path.join(data_dir, image_class)))
    count = 0
    print(image_class)
    print("#############################################")
    for image in os.listdir(os.path.join(data_dir, image_class)):
        #print(image)
        img = tf.keras.utils.load_img(
            os.path.join(data_dir, image_class, image), target_size=(img_height, img_width)
        )
        #img = image
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        if format(class_names[np.argmax(score)]) == image_class:
            count += 1

        #for i in range(5):
        #    print(score[i])

        #print(
        #    "This image most likely belongs to {} with a {:.2f} percent confidence."
        #    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    scores.append([count/full, image_class])
#print(f"{count/full}% correct in {image_class} ")
'''
