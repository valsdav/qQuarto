from keras import backend as K
import numpy as np


def prioritization_loss(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred))
