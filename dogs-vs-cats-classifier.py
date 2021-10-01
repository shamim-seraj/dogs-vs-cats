# import
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model, load_model

# check if GPU available
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
device_lib.list_local_devices()

# load the data
train_path = '/home/shuvornb/Desktop/Data Mining/dogs-vs-cats/train'
valid_path = '/home/shuvornb/Desktop/Data Mining/dogs-vs-cats/valid'

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,
                                                                                             target_size=(224, 224),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=10)
valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                             target_size=(224, 224),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=5)

# build the model
model = Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

# compile the model with optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# This callback will stop the training when there is no improvement in the loss for three consecutive epochs.
early_stopping = EarlyStopping(monitor='loss', patience=3)

# train the model
history = model.fit(
    train_batches,
    epochs=50,
    validation_data=valid_batches,
    steps_per_epoch=2000,
    validation_steps=1000,
    callbacks=[early_stopping]
)

# Save the model
filepath = './saved_model'
save_model(model, filepath)
