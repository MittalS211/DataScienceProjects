import os
import time
import cv2
import keras
from keras.optimizers import RMSprop, Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
# implement class to store Epoch runtimes#

def load_images(data_path, categories, 
                IMG_SIZE = 50, augment=None, 
                grayscale=True):
    data = list()
    target = list()
    errors = 0
    for i, category in enumerate(categories): 
        filepaths = []
        images_dir = data_path+category
        for root, dirs, files in os.walk(images_dir):
             for file in files:
                filepaths.append((os.path.join(root, file)))

        filepaths = [path for path in filepaths if path.endswith('jpg')]
        
        for path in filepaths:
            try:
                img_path = path

                if grayscale ==True:
                    img_data = cv2.imread(img_path, 0)
                
                else:
                    img_data = cv2.imread(img_path, 1) # read image in color
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB) # convert to RGB color mapping


                img_data = cv2.resize(img_data, (IMG_SIZE,IMG_SIZE))
                #print(img_data.shape)
                
                # normalize pixels 
                data.append(img_data)
                target.append(i)
                
                if augment:
                    data.append(np.fliplr(img_data))
                    target.append(i)
                
            except Exception as e:
                errors +=1 # track errors
                continue
    print("{} images unable to be loaded.".format(errors))
    return (data,target)

def process_data(X, y, n_classes,  IMG_SIZE=50, grayscale=True):
    
    new_X = np.array(X) / 255 #normalize pixels
    
    if grayscale==True:
        
        new_X = new_X.reshape(-1,IMG_SIZE,IMG_SIZE,1)
        
    else:
        new_X = new_X.reshape(-1,IMG_SIZE,IMG_SIZE,3) #format data with color channels

    
    new_y = keras.utils.np_utils.to_categorical(y, n_classes)
    
    return (new_X, new_y)

def compile_and_fit(model, fit_params, loss= 'categorical_crossentropy', opt=RMSprop(), metrics=['accuracy'], log_times=False):
    """function to compile and fit neural networks in one step, 
    
       model: keras model to compile 
       fit_params: dictionary of model fit parameters mapped to dict keys: ['X_train', 'y_train', 'batch_size', 'epochs', 'validation_data']
       loss: loss function
       opt: optimizer
       metrics: model metrics
       Log_times: boolean, if True function returns list of runtimes of each epoch in the model, depends on *TimeHistory Class* default False
       """
    
    model_copy = model
    model_copy.compile(loss=loss,
              optimizer=opt,
              metrics=metrics)
    if log_times==False:
        model_history = model_copy.fit(fit_params["X_train"], fit_params["y_train"], batch_size = fit_params["batch_size"], 
                  epochs = fit_params["epochs"], 
               validation_data = fit_params["validation_data"], verbose = 2)
        return model_history
    
    else:
        time_callback = TimeHistory()
        model_history = model_copy.fit(fit_params["X_train"], fit_params["y_train"], batch_size = fit_params["batch_size"], 
              callbacks = [time_callback], epochs = fit_params["epochs"], 
           validation_data = fit_params["validation_data"], verbose = 2)
        return model_history, time_callback.times

# store model history as csv for easy retrieval
def save_model_history(model_callback,filename, subdir=None, times=None):   
    df = pd.DataFrame.from_dict(model_callback.history)

    if times:
        df['epoch_runtime'] = times
    if subdir:
        filepath ='model_history/{}/{}.csv'.format(subdir, filename)
    else:
        filepath = 'model_history/{}.csv'.format(filename)
    df.to_csv(filepath)
    print('\nmodel history saved at {}'.format(filepath))
    return filepath


print("Helper functions loaded successfully!")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

## pipeline to run list of one-to-one genre comparison over a number of models and save results to a csv 

def run_save_models(models, model_names, labels, params, filepath, IMG_SIZE=50, aug=False, grayscale=True):
    
    if aug == False:
        aug = None
    
    if grayscale==True:
        X, y = load_images(filepath, categories=labels, IMG_SIZE = IMG_SIZE, augment = aug, grayscale=True)
        X, y = process_data(X, y, n_classes=len(labels), IMG_SIZE=IMG_SIZE, grayscale=True)
    else:
        X, y = load_images(filepath, categories=labels, IMG_SIZE = IMG_SIZE, augment = aug, grayscale=False)
        X, y = process_data(X, y, n_classes=len(labels), IMG_SIZE=IMG_SIZE, grayscale=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 0)
    
    
    params['X_train'] = X_train
    params['y_train'] = y_train
    params['validation_data'] = (X_test, y_test)
    files = []
    cms = []
    for model, model_name in zip(models, model_names):

        model_history, runtimes = compile_and_fit(model, params, opt=Adam(),log_times=True)
        print(model.summary())
        if model_name not in os.listdir('model_history/'):
            os.mkdir('model_history/{}'.format(model_name))

        files.append(save_model_history(model_history, times=runtimes,subdir=model_name\
                                         , filename="{} or {}".format(labels[0],labels[1])))
        
        #generate confusion matrix
        y_pred = model.predict_classes(X_test)

        cm = confusion_matrix(y_test[:, 1], y_pred)
        
        cms.append(cm)
    return files, cms

print("pipeline ready!")
