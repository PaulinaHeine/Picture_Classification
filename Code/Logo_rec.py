##########
#Difference between cisco & Sun
##########


# Setup
import tensorflow as tf
import os


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Import & Filter images
import cv2
import imghdr

data_dir = "/Users/paulinaheine/Codes//Logos"
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

ds_store_file_location = '/Users/paulinaheine/Codes/Logos/.DS_store'
if os.path.isfile(ds_store_file_location):
    os.remove(ds_store_file_location)


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        #label = assign_label(image, image_class)
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

data = tf.keras.utils.image_dataset_from_directory('/Users/paulinaheine/Codes//Logos',labels='inferred', color_mode= "rgb" )
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

'''
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
'''


# Scale Data
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()

# Split Data
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Train
logdir = '/Users/paulinaheine/Codes//Logos'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Plot Performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Evaluate
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())

import cv2

img = cv2.imread('/Users/paulinaheine/Codes//Logos/cisco/Luzon_Cisco_Transport2C_Inc._logo.jpg')
plt.imshow(img)
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize / 255, 0))
yhat
y_classes = yhat.argmax(axis=-1)

if yhat > 0.5:
    print(f'Predicted class is Sun')
else:
    print(f'Predicted class is Cisco')


img = cv2.imread('/Users/paulinaheine/Codes//Logos/cisco/Luzon_Cisco_Transport2C_Inc._logo.jpg')
resize = tf.image.resize(img, (256, 256))

yhat = model.predict(np.expand_dims(resize / 255, 0))
yhat
y_classes = yhat.argmax(axis=-1)

if yhat > 0.5:
    print(f'Predicted class is Sun')
else:
    print(f'Predicted class is Cisco')
