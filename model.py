import tensorflow as tf
import numpy as np
import config
from keras import layers, optimizers
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Adamax
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dense, LSTM, Dropout, Flatten, Input, Conv1D, MaxPooling1D
from keras.activations import relu, elu, softmax, selu
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

def get_model(n_classes):
    # Set seed to get same results
    np.random.seed(121)
    # Define model as Sequential class
    model_cnn_selu = Sequential()

    # Block 1

    # first layer: convolutional layer with filter size of 2 and strid of 2, due to 
    # the features having shape divisable by 2. selu shown to work better than relu
    # and leaky relu without CNN.
    # Max Pool to get only the important features
    # add dropout to combat overfitting. rate = 10% of neurons to drop
    model_cnn_selu.add(Conv1D(128, (2), strides=(2), padding='same', 
                        input_shape = (1,192), activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))
    model_cnn_selu.add(Dropout(rate = 0.1))

    # Block 2
    model_cnn_selu.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))
    model_cnn_selu.add(Dropout(rate = 0.1))

    # Block 3
    model_cnn_selu.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
    model_cnn_selu.add(Conv1D(512, (2), strides=(2), padding='same',activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))

    # Block 4
    model_cnn_selu.add(Conv1D(512, (2), strides=(2), padding='same', activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))
    model_cnn_selu.add(Dropout(rate = 0.1))

    # Block 5
    model_cnn_selu.add(Conv1D(512, (2), strides=(2), padding='same',activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))

    # Block 7
    model_cnn_selu.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))

    # Block 8 
    model_cnn_selu.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))
    model_cnn_selu.add(Dropout(rate = 0.1))

    # Block 9
    model_cnn_selu.add(Conv1D(256, (2), strides=(2), padding='same', activation= 'selu'))
    model_cnn_selu.add(MaxPooling1D((2), strides=(2), padding='same'))

    # Output Block
    model_cnn_selu.add(layers.Flatten())
    model_cnn_selu.add(Dense(n_classes, activation = 'softmax'))

    return model_cnn_selu

def load_model():
    # Read in model
    json_file = open(config.MODEL_PATH, 'r')
    loaded_model_json = json_file.read()

    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(config.MODEL_WEIGHTS_PATH)
    print("Loaded model into notebook")
    return loaded_model