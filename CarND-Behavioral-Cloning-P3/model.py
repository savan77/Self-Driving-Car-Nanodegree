#import dependecies
from keras.models import Sequential
from keras.layers import Convolution2D, Cropping2D, Flatten, Dense, Lambda
from preprocess import data_generator, get_data


#nvidia model from the course
def steer_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #normalization 
    model.add(Cropping2D(cropping=((50,20), (0,0))))  #crop interesting part
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#prepare data
train_data, valid_data = get_data("dataset/")
#create generator for keras
train_generator = data_generator(train_data)
valid_generator = data_generator(valid_data)

#initialize model
model = steer_model()
#since this is regression model use Mean Squared Error
model.compile(loss='mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_data), validation_data=valid_generator, \
                 nb_val_samples=len(valid_data), nb_epoch=3, verbose=1)

#save the model
model.save('model.h5')

#print the losses
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])