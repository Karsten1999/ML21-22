from random import choices
import numpy as np


def prediction_vector_pick_highest(y_pred, zero_bias: float = 1):
    """"
    Function that given a prediction vector picks the highest probability element and picks that one

    :param y_pred: the predicted results
    :param zero_bias: a value by which the chance of zero is multiplied to ensure less zeroes appear

    :returns score of the prediction
    """
    y_prediction = []
    for yp in y_pred:
        yp[0] = yp[0] * zero_bias
        yp[yp < max(yp)] = 0
        yp[yp == max(yp)] = 1
        y_prediction.append(yp)
    return y_prediction


def prediction_vector_pick_stochastically(y_pred, zero_bias: float = 1):
    """"
    Function that given a prediction vector picks with the probability elements of the vector, after they are
    normalised

    :param y_pred: the predicted results

    :returns vector with a prediction for the given y_pred
    """
    y_prediction = []
    # Important assumption here is that each element of y_pred is the same size, should happen all the time, but can
    # be a source of error
    indices = np.arange(0,len(y_pred[0]),1)
    # Used to create output
    zeroes = np.zeros(len(y_pred[0]))
    for yp in y_pred:
        # Zero bias
        yp[0] = yp[0] * zero_bias
        # Setting negative values to zero
        yp[yp<0]=0
        # Normalising
        yp = yp/sum(yp)

        index = choices(indices, yp)
        result = zeroes.copy()
        result[index] = 1

        y_prediction.append(result)

    return y_prediction

def vector_to_note(min_note: int, vector) -> int:
    """"
    Function to return a note number from a vectorised note

    :param min_note: minumum note value
    :param vector: the note vector

    :returns key number of the note
    """
    return int(np.where(vector==1)[0][0] - min_note)