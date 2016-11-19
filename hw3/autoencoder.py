## ===============================
## import numpy and keras packages
## ===============================
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
## For autoencoder
from keras.layers import Input, Dense
from keras.models import Model
data_path = sys.argv[1]
model_name = sys.argv[2]

## ==============================
## Load HW3 dataset and normalize
## ==============================
import pickle
all_label = pickle.load(open(data_path+'all_label.p','rb'))
x_train_HW3 = np.array(all_label, np.uint8).reshape(5000, 3072)
y_train_HW3 = np_utils.to_categorical([i for i in range(10) for j in range(500)], 10)
all_unlabel = pickle.load(open(data_path+'all_unlabel.p','rb'))
x_unlabel = np.array(all_unlabel, np.uint8).reshape(45000, 3072)
test = pickle.load(open(data_path+'test.p','rb'))
x_test_HW3 = np.array(test['data'], np.uint8).reshape(10000, 3072)
x_train_HW3 = x_train_HW3.astype('float32')
x_test_HW3 = x_test_HW3.astype('float32')
x_unlabel = x_unlabel.astype('float32')
x_train_HW3 /= 255
x_test_HW3 /= 255
x_unlabel /= 255

# ===========
# Autoencoder
# ===========
from keras.callbacks import EarlyStopping
early_stopping1 = EarlyStopping(monitor = 'loss', patience = 3)

# Training the 1st layer:
x_train_tmp = np.concatenate((x_train_HW3,x_unlabel), axis = 0)
input_img1 = Input(shape=(3072,))
encoded1 = Dense(512, activation='relu')(input_img1)
decoded1 = Dense(3072, activation='sigmoid')(encoded1)
autoencoder1 = Model(input=input_img1, output=decoded1)
encoder1 = Model(input=input_img1, output=encoded1)
decoded_input1 = Input(shape=(512,))
decoder_layer1 = autoencoder1.layers[-1]
decoder1 = Model(input=decoded_input1, output=decoder_layer1(decoded_input1))
autoencoder1.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder1.fit(x_train_tmp, x_train_tmp,
                 nb_epoch=50,
                 batch_size=64,
                 shuffle=True,
                 callbacks=[early_stopping1])
# Training the 2nd layer:
x_train_tmp = encoder1.predict(x_train_tmp)
input_img2 = Input(shape=(512,))
encoded2 = Dense(256, activation='relu')(input_img2)
decoded2 = Dense(512, activation='sigmoid')(encoded2)
autoencoder2 = Model(input=input_img2, output=decoded2)
encoder2 = Model(input=input_img2, output=encoded2)
decoded_input2 = Input(shape=(256,))
decoder_layer2 = autoencoder2.layers[-1]
decoder2 = Model(input=decoded_input2, output=decoder_layer2(decoded_input2))
autoencoder2.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder2.fit(x_train_tmp, x_train_tmp,
                 nb_epoch=100,
                 batch_size=64,
                 shuffle=True,
                 callbacks=[early_stopping1])
# Training the 3rd layer:
x_train_tmp = encoder2.predict(x_train_tmp)
input_img3 = Input(shape=(256,))
encoded3 = Dense(64, activation='relu')(input_img3)
decoded3 = Dense(256, activation='sigmoid')(encoded3)
autoencoder3 = Model(input=input_img3, output=decoded3)
encoder3 = Model(input=input_img3, output=encoded3)
decoded_input3 = Input(shape=(64,))
decoder_layer3 = autoencoder3.layers[-1]
decoder3 = Model(input=decoded_input3, output=decoder_layer3(decoded_input3))
autoencoder3.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder3.fit(x_train_tmp, x_train_tmp,
                 nb_epoch=100,
                 batch_size=64,
                 shuffle=True,
                 callbacks=[early_stopping1])

# ============
# Fine-turning
# ============
from sklearn.utils import shuffle
x_train_HW3, y_train_HW3 = shuffle(x_train_HW3, y_train_HW3, random_state = 1002)

# Fine-turning
input_img = Input(shape=(3072,))
layer1 = encoder1(input_img)
layer2 = encoder2(layer1)
layer3 = encoder3(layer2)
# output_layer = Dense(10, activation='softmax')(layer1)
output_layer = Dense(10, activation='softmax')(layer3)
model = Model(input=input_img, output=output_layer)
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
# adam_ = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
	          optimizer=sgd,
	          metrics=['accuracy'])

hist = model.fit(x_train_HW3, y_train_HW3,
                 batch_size=16, nb_epoch=150,
                 shuffle = True,
                 validation_split = 0.2)

model.save(model_name)