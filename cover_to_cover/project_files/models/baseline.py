from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

IMG_SIZE = 50

n_classes = 2
# Start with a simple sequential model
model = Sequential()

# Add dense layers to create a fully connected MLP
# Note that we specify an input shape for the first layer, but only the first layer.
# Relu is the activation function used
model.add(Dense(128, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))

# Dropout layers remove features and fight overfitting
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))


# reduce dimensions to match out shape to target
model.add(Flatten())

#model.add(Flatten())
# End with a number of units equal to the number of classes we have for our outcome
model.add(Dense(n_classes, activation='softmax'))



model_name = 'baseline'
bold_flag ='\033[1m'

print(bold_flag+"Multilayer Perceptron - Baselise")

model.summary()

