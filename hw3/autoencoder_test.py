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
output_name = sys.argv[3]

## ================
## Load HW3 dataset
## ================
import pickle
test = pickle.load(open(data_path+'test.p','rb'))
x_test_HW3 = np.array(test['data'], np.uint8).reshape(10000, 3072)
x_test_HW3 = x_test_HW3.astype('float32')
x_test_HW3 /= 255

from keras.models import load_model
model = load_model(model_name)
pred = model.predict(x_test_HW3)
np.savetxt(output_name, np.dstack((np.arange(pred.size),pred))[0],fmt="%d,%d", header = "ID,class", comments='')
