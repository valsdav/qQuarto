from keras import backend as K
import numpy as np


def prioritization_loss(weights):
    def loss(y_true, y_pred):
        print(y_true, y_pred, weights)
        return K.sum(K.square(y_true - y_pred) * weights)
    return loss
