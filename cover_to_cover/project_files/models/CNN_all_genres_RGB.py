from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import regularizers
n_classes = 23

IMG_SIZE = 50

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),padding = 'Same',
                 activation='relu',
                 input_shape=(IMG_SIZE, IMG_SIZE, 3)))
#model.add(Dropout(0.25))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), strides=(2,2), padding = 'Same', activation='relu'))
model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.01), padding = 'Same', activation='relu'))
#model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l1(0.01), padding = 'Same', activation='relu'))
#model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.summary()