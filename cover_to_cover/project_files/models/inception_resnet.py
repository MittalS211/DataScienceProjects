from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
n_classes = int(input('Enter number of labels '))
IMG_SIZE = int(input('Set Image Size - min 75 '))

base_model = InceptionResNetV2(
    weights = 'imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE, 3))


for layer in base_model.layers[:]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in base_model.layers:
    print(layer, layer.trainable)
base_model.summary()

model = Sequential()
 
# Add the vgg convolutional base model
model.add(base_model)
 
# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
