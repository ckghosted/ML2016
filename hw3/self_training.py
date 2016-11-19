## ===============================
## import numpy and keras packages
## ===============================
import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
data_path = sys.argv[1]
model_name = sys.argv[2]

## ================
## Load HW3 dataset
## ================
import pickle
all_label = pickle.load(open(data_path+'all_label.p','rb'))
x_train_HW3 = np.array(all_label, np.uint8).reshape(5000, 3, 32, 32)
y_train_HW3 = np_utils.to_categorical([i for i in range(10) for j in range(500)], 10)
test = pickle.load(open(data_path+'test.p','rb'))
x_test_HW3 = np.array(test['data'], np.uint8).reshape(10000, 3, 32, 32)
all_unlabel = pickle.load(open(data_path+'all_unlabel.p','rb'))
x_unlabel = np.array(all_unlabel, np.uint8).reshape(45000, 3, 32, 32)

## =============
## Normalization
## =============
x_train_HW3 = x_train_HW3.astype('float32')
x_test_HW3 = x_test_HW3.astype('float32')
x_unlabel = x_unlabel.astype('float32')
x_train_HW3 /= 255
x_test_HW3 /= 255
x_unlabel /= 255
## Append the testing data to make more unlabeled data
x_unlabel_more = np.concatenate((x_unlabel, x_test_HW3), axis = 0)

## =============
## Self-training
## =============
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## Initialze a boolean array indicating if each unlabeled data sample has qualified pseudo label
# is_labeled = np.array([False], np.bool).repeat(x_unlabel_more.shape[0])

## Repeat until when???
# model_temp = load_model("model_super1_000.h5")
# x_train_temp = x_train_HW3
# y_train_temp = y_train_HW3
x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(x_train_HW3, y_train_HW3, test_size = 0.2)
x_unlabel_temp = x_unlabel_more
hist = list()
count = 1
while(x_unlabel_temp.shape[0] > 0 and count < 9):
    model_temp = Sequential()
    model_temp.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = x_train_HW3.shape[1:]))
    model_temp.add(Activation('relu'))
    model_temp.add(Convolution2D(32, 3, 3))
    model_temp.add(Activation('relu'))
    model_temp.add(MaxPooling2D(pool_size=(2, 2)))
    model_temp.add(Dropout(0.25))
    model_temp.add(Convolution2D(64, 3, 3, border_mode='same'))
    model_temp.add(Activation('relu'))
    model_temp.add(Convolution2D(64, 3, 3))
    model_temp.add(Activation('relu'))
    model_temp.add(MaxPooling2D(pool_size=(2, 2)))
    model_temp.add(Dropout(0.25))
    model_temp.add(Flatten())
    model_temp.add(Dense(512))
    model_temp.add(Activation('relu'))
    model_temp.add(Dropout(0.5))
    model_temp.add(Dense(10)) ## nb_class = 10
    model_temp.add(Activation('softmax'))
    
    ## adaptive learning rate and batch size
    adaLearningRate = 0.017 * 4000 / x_train_temp.shape[0]
    adaBatchSize = 64 * x_train_temp.shape[0] / 4000
    print adaLearningRate, adaBatchSize
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2+count)
    
    sgd = SGD(lr=adaLearningRate, decay=1e-6, momentum=0.9, nesterov=True)
    model_temp.compile(loss='categorical_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy'])
    x_train_temp_shuffle, y_train_temp_shuffle = shuffle(x_train_temp, y_train_temp)
    
    hist_temp = model_temp.fit(x_train_temp_shuffle, y_train_temp_shuffle,
                               batch_size=adaBatchSize,nb_epoch=200,
#                                validation_split=0.2,
                               validation_data=(x_test_temp, y_test_temp),
                               callbacks=[early_stopping])
    
    if len(hist) > 1 and max(hist_temp.history.get('val_acc')) < max(hist[-1].history.get('val_acc')):
        print 'STOP!!!!!!!!!!!!!!!!!!!!!'
        model_final = Sequential()
        model_final.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = x_train_HW3.shape[1:]))
        model_final.add(Activation('relu'))
        model_final.add(Convolution2D(32, 3, 3))
        model_final.add(Activation('relu'))
        model_final.add(MaxPooling2D(pool_size=(2, 2)))
        model_final.add(Dropout(0.25))
        model_final.add(Convolution2D(64, 3, 3, border_mode='same'))
        model_final.add(Activation('relu'))
        model_final.add(Convolution2D(64, 3, 3))
        model_final.add(Activation('relu'))
        model_final.add(MaxPooling2D(pool_size=(2, 2)))
        model_final.add(Dropout(0.25))
        model_final.add(Flatten())
        model_final.add(Dense(512))
        model_final.add(Activation('relu'))
        model_final.add(Dropout(0.5))
        model_final.add(Dense(10)) ## nb_class = 10
        model_final.add(Activation('softmax'))

        sgd = SGD(lr=adaLearningRate, decay=1e-6, momentum=0.9, nesterov=True)
        model_final.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])
        x_train_temp_shuffle, y_train_temp_shuffle = shuffle(x_train_temp, y_train_temp)
        x_train_final = np.concatenate((x_train_temp, x_test_temp), axis = 0)
        y_train_final = np.concatenate((y_train_temp, y_test_temp), axis = 0)
        hist_temp = model_final.fit(x_train_final, y_train_final,
                                    batch_size=adaBatchSize,nb_epoch=200,
                                    validation_split=0.2,
                                    callbacks=[early_stopping])
        model_final.save(model_name)
        break
    hist.append(hist_temp)
    ## Predict (probability)
#     pred_temp = model_temp.predict(x_unlabel_temp,batch_size=x_unlabel_temp.shape[0])
    pred_temp = model_temp.predict(x_unlabel_temp)
    ## For each testing sample, find its max probability among all classes
    ## ([0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.4] ==> 0.4)
    pred_temp_max = np.array([max(pred_temp[i]) for i in xrange(x_unlabel_temp.shape[0])])
    ## Find the 5000-th probability among all pred_temp_max
    ## ([0.7, 0.65, 0.8, 0.9, 0.77, 0.68] ==> find the 3rd ==> 0.77)
    threshold_temp = pred_temp_max[pred_temp_max.argsort()[-2500:][0]]
    threshold_temp = 0.99
    ## Re-define temp training and unlabeled data for next iteration
    x_train_temp = np.concatenate((x_train_temp, x_unlabel_temp[pred_temp_max >= threshold_temp]), axis = 0)
    pred_temp_qualified = pred_temp[pred_temp_max >= threshold_temp]
    y_train_temp = np.concatenate((y_train_temp, np_utils.to_categorical([pred_temp_qualified[i].argmax() for i in range(pred_temp_qualified.shape[0])], 10)), axis = 0)
    x_unlabel_temp = x_unlabel_temp[pred_temp_max < threshold_temp]
    ## Make sure x_unlabel_temp will be smaller and smaller
    print x_unlabel_temp.shape
    ## Prevent infinite loop
    count = count + 1

