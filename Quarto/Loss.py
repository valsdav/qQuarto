from keras import backend as K
import numpy as np

weights = np.ones(32)

def prioritization_loss(y_true, y_pred):
    global weights
    return K.sum(K.square(y_true - y_pred) * weights)
