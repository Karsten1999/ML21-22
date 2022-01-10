from sklearn.metrics import r2_score
from random import choices
import numpy as np


def prediction_vector_pick_highest(y_true, y_pred):
    """"
    Function that given a prediction vector picks the highest probability element and picks that one

    :param y_true: the correct results
    :param y_pred: the predicted results

    :returns score of the prediction
    """
    y_prediction = []
    for yp in y_pred:
        yp[yp < max(yp)] = 0
        yp[yp == max(yp)] = 1
        y_prediction.append(yp)
    return r2_score(y_true, y_prediction)


def prediction_vector_pick_stochastically(y_true, y_pred):
    """"
    Function that given a prediction vector picks with the probability elements of the vector, after they are
    normalised

    :param y_true: the correct results
    :param y_pred: the predicted results

    :returns score of the prediction
    """
    y_prediction = []
    # Important assumption here is that each element of y_pred is the same size, should happen all the time, but can
    # be a source of error
    indices = np.arange(0,len(y_pred[0]),1)
    # Used to create output
    zeroes = np.zeros(len(y_pred[0]))
    for yp in y_pred:
        # Normalising
        yp = yp/sum(yp)

        index = choices(indices, yp)
        result = zeroes.copy()
        result[index] = 1

        y_prediction.append(result)

    return r2_score(y_true, y_prediction)

