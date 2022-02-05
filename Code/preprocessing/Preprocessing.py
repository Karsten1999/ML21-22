import numpy as np
from sklearn.model_selection import train_test_split


def Transform_into_vector(data: np.array, min_value: int = None, max_value: int = None):
    """
    :param data: the input voice
    :param min_value: optional minimum value for the vector
    :param max_value: optional maxmimum value for the vector

    :return: matrix with note vectors
    """
    data = data.astype("int")

    if data.ndim == 1:
        if min_value and not max_value:
            if min_value > min(data):
                raise ValueError("Min_value should at most be the size of the smallest note")
        elif not min_value and max_value:
            if max_value < max(data):
                raise ValueError("Max_value should at least be the size of the largest note")
        elif min_value and max_value:
            if max_value - min_value < max(data) - min(data):
                raise ValueError("Max_value and min_value should at least be the size of the largest note and "
                                 "respectively at most the smallest note")
        if not min_value:
            min_value = int(min(data[data>0]))
        if not max_value:
            max_value = int(max(data))

        # Create zero matrix which has columns equal to the note vector and is as long as the data
        matrix = np.zeros((data.size, max_value - min_value + 2))
        # Finding the positions in each note vector
        positions = data - min_value + 1
        # Since 0 will be the first element and the subsequent elements are all move up (e.g. note 23 is not on the 23rd
        # position, but on the second
        # and thus isnt the exact position we have to manually change that
        positions[positions<0] = 0

        matrix[np.arange(0, data.size), positions]=1
        return matrix
    else:
        raise ValueError("Data dimension should be 1, not implemented yet for 2, but you can just manually loop.")


def Split_rolling_window(data, vector, test_size: float = None, window_size: int = 16, train = True, output_size = 1):
    """
    Function to split the data into a training and test set

    :param data: the data in terms of key note, used to make X
    :param vector: the data in terms of a vector, used to make y
    :param window_size: size of the training windows
    :param train: whether to make a test and training set or just return the complete X and y
    :param output_size: size of the y-vectors
    :return:
    """

    X = []
    y = []

    # Creating X and Y by using a rolling window
    if output_size==1:
        for i in range(0, len(vector) - window_size):
            X.append(data[i:i+window_size])
            y.append(vector[i+window_size])
    else:
        for i in range(0, len(vector) - window_size - output_size):
            X.append(data[i:i + window_size])
            y.append(vector[i + window_size: i + window_size + output_size].flatten())
    X = np.array(X)
    y = np.array(y)
    if not train:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def transform_into_difference(data: np.array):
    """"
    Function to transform the data to a form that compares the difference with the previous note

    :param data: the data in terms of key note values

    :returns: data in terms of difference with the previous note
    """

    return data[1::] - data[0:-1]


if __name__ == "__main__":
    print(Transform_into_vector(np.array([0,25,27,27,27,26])))
