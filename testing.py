from threading import Thread
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Input())
