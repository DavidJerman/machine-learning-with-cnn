# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Keras librararies
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Lambda
from keras.layers import BatchNormalization

image_size = 32
path = "C:\\Users\\[name]\\ML Datasets\\Characters"

# # Initializing the CNN
# classifier = Sequential()

# x = Input(shape=(image_size, image_size, 3))
# a = Convolution2D(64, (3, 3), activation="relu")(x)
# y = Flatten()(a)
# z = Dense(units=16, activation="softmax")(a)
# classifier = Model(inputs=x, outputs=z)

# # The first param is the size and number of convolutions
# # Step 1 - Convolution
# classifier.add(Convolution2D(32, (3, 3), input_shape=(image_size, image_size, 3), activation="relu"))
# classifier.add(Convolution2D(32, (3, 3), activation="relu"))
# # classifier.add(LReLU(alpha=0.3))
#
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.4))
#
# # Step 1 - Convolution
# classifier.add(Convolution2D(64, (3, 3), activation="relu"))
# classifier.add(Convolution2D(64, (3, 3), activation="relu"))
# classifier.add(Convolution2D(64, (3, 3), activation="relu"))
# # classifier.add(LReLU(alpha=0.3))
#
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.4))
#
# # Step 3 - Flattening
# classifier.add(Flatten())
#
# # Step 4 - Full connection - The hidden layers
# # classifier.add(Dense(units = 256, activation = "relu"))
#
# # Second experimental layer
# classifier.add(Dense(units=186, activation="relu"))
#
# # The output layer
# classifier.add(Dense(units=62, activation="softmax"))
#
# adam = Adam(lr=0.0001)
#
# # Compiling the CNN
# # categorical_crossentropy if we had more than 2 output nodes
# classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Super efficient network
classifier = Sequential([
    Convolution2D(32,(3,3), activation='relu', input_shape=(image_size,image_size,1)),
    BatchNormalization(),
    Convolution2D(32,(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Convolution2D(64,(3,3), activation='relu'),
    BatchNormalization(),
    Convolution2D(64,(3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(62, activation='softmax')
])
classifier.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
classifier.summary()

# Part 2 - Fitting the CNN to the images
# We use image augmentation to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(path + '/train',
                                                 target_size=(image_size, image_size),
                                                 batch_size=64,
                                                 class_mode='categorical',
                                                 color_mode="grayscale")

test_set = test_datagen.flow_from_directory(path + '/test',
                                            target_size=(image_size, image_size),
                                            batch_size=64,
                                            class_mode='categorical',
                                            color_mode="grayscale")

print(training_set.__len__())

classifier.fit_generator(training_set,
                         steps_per_epoch=training_set.__len__(),
                         epochs=2,
                         validation_data=test_set,
                         validation_steps=test_set.__len__() / 8)

classifier.save("./models/model_final_plus_plus.h5")

# classifier.fit_generator(training_set,
#                          steps_per_epoch=training_set.__len__(),
#                          epochs=20,
#                          validation_data=test_set,
#                          validation_steps=test_set.__len__() / 8)
#
# classifier.save("./models/model_5_3.h5")
