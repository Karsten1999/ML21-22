import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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
            min_value = int(min(data))
        if not max_value:
            max_value = int(max(data))

        # Create zero matrix which has columns equal to the note vector and is as long as the data
        matrix = np.zeros((data.size, max_value - min_value + 1))
        # Finding the positions in each note vector
        positions = data - min_value
        matrix[np.arange(0,data.size),positions]=1
        return matrix
    else:
        raise ValueError("Data dimension should be 1, not implemented yet for 2, but you can just manually loop.")


def Split_rolling_window(data, vector, test_size: float = None, window_size: int = 16):
    """
    Function to split the data into a training and test set

    :param data: the data in terms of key note, used to make X
    :param vector: the data in terms of a vector, used to make y
    :param n_splits: number of splits to make, if more than 1 cross-validation is used to score it
    :param window_size: size of the training windows
    :return:
    """

    X = []
    y = []

    # Creating X and Y by using a rolling window
    for i in range(0, len(vector) - window_size):
        X.append(data[i:i+window_size])
        y.append(vector[i+window_size])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    #Split_rolling_window([[0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]],window_size = 2)


    voice1 = np.loadtxt("F.txt").T[0]
    vector = Transform_into_vector(voice1)
    X_train, X_test, y_train, y_test = Split_rolling_window(voice1, vector, window_size=100)
    reg = LinearRegression().fit(X_train, y_train)
    score = reg.score(X_test, y_test)

    print(score)
