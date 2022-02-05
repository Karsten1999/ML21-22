from random import choices
import numpy as np


class AllCompositions:
    """"
    Class to store TraversingComposition
    """
    def __init__(self):
        self.compositions = []

    def __repr__(self):
        for composition in self.compositions:
            print("Composition:", composition.composition, " Chance:", composition.chance)

    def add_new(self, composition):
        """"
        Add a new composition
        """
        self.compositions.append(composition)


class TraversingComposition:
    """"
    Class to store vectors for the function prediction_vector_traverse
    """
    def __init__(self, chance: float, composition):
        """"
        :param chance: current chance of path
        :param composition: the current combination of note vectors
        """
        self.chance = chance
        self.composition = composition

    def __len__(self):
        """"
        Returns length of the composition
        """
        return len(self.composition)

    def add_new(self, note):
        """"
        Add a new note to the composition
        """
        self.composition.append(note)

    def copy(self):
        """"
        Create a copy of this object
        """
        return TraversingComposition(self.chance, self.composition.copy())


def prediction_vector_pick_highest(y_pred, zero_bias: float = 1):
    """"
    Function that given a prediction vector picks the highest probability element and picks that one

    :param y_pred: the predicted results
    :param zero_bias: a value by which the chance of zero is multiplied to ensure less zeroes appear

    :returns score of the prediction
    """
    if y_pred.ndim == 1:
        y_pred[0] = y_pred[0] * zero_bias
        y_pred[y_pred < max(y_pred)] = 0
        y_pred[y_pred == max(y_pred)] = 1
        return y_pred
    if y_pred.ndim == 2:
        y_prediction = []
        for yp in y_pred:
            yp[0] = yp[0] * zero_bias
            yp[yp < max(yp)] = 0
            yp[yp == max(yp)] = 1
            y_prediction.append(yp)
        return y_prediction


def prediction_vector_pick_stochastically(y_pred, zero_bias: float = 1, min_chance: float = None):
    """"
    Function that given a prediction vector picks with the probability elements of the vector, after they are
    normalised

    :param y_pred: the predicted results
    :param zero_bias: a value by which the chance of zero is multiplied to ensure less zeroes appear
    :param min_chance: minimum chance required

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
        # If there is a min_chance we renormalise again
        if min_chance:
            yp[yp < min_chance] = 0
            yp = yp / sum(yp)

        index = choices(indices, yp)
        result = zeroes.copy()
        result[index] = 1

        y_prediction.append(result)

    return y_prediction


def prediction_vector_traverse(X, reg, min_note, n: int = 3, limit_chance: float = 0.1, prediction_length: int = 32):
    """"
    Function that picks the top n prediction notes and then predicts future notes using that, it does this recursively,
    the process will be stopped when the prediction length has been reached.

    :param X: the x-vector of the data, assumes window size is already included here
    :param reg: the linear regression fit
    :param min_note: the minimum note in the voice
    :param n: the amount of prediction notes to pick
    :param limit_chance: the minimum chance that a note is the next note
    :param prediction_length: number of notes to predict

    :returns y_out: the prediction vector of length prediction_length
    """
    # Output will be a data class with all possible compositions above a certain chance
    y_out = AllCompositions()

    y_pred = reg.predict(X)[0]

    # Setting negative values to zero
    y_pred[y_pred < 0] = 0
    # Normalising
    y_pred = y_pred / sum(y_pred)

    # Picking top n  with highest chances
    highest_indices = y_pred.argsort()[-n::]

    # Used to create output
    zeroes = np.zeros(len(y_pred))
    for index in highest_indices:
        # Creating resulting vector which has zeroes everywhere except one spot
        result = zeroes.copy()
        result[index] = 1

        chance = y_pred[index]

        new_composition = TraversingComposition(chance, [vector_to_note(min_note, result)])

        prediction_vector_traverse_recurse(X=X, reg=reg, min_note=min_note, current_composition=new_composition,
                                           allcompositions=y_out, n=n,
                                           limit_chance=limit_chance * limit_chance,
                                           global_limit_chance=limit_chance, prediction_length=prediction_length)
    return y_out


def prediction_vector_traverse_recurse(X, reg, min_note, current_composition, allcompositions, n: int = 3
                                       , limit_chance: float = 0.1, global_limit_chance: float = 0.1,
                                       prediction_length: int = 32):
    """"
    Recursive function working with prediction_vector_traverse

    :param X: the x-vector of the data, assumes window size is already included here
    :param reg: the linear regression fit
    :param min_note: the minimum note in the voice
    :param current_composition: the current composition object
    :param allcompositions: the collection of compositions objects
    :param n: the amount of prediction notes to pick
    :param limit_chance: the minimum chance that a note is the next note
    :param global_limit_chance: the limit chance for the current note
    :param prediction_length: number of notes to predict

    :returns y_out: the prediction vector of length prediction_length
    """
    # X used to predict, should include appended current composition and shift forwards with the length of the current
    # composition. i.e. if the current composition is 4 long, we should shift the window forwards by 4
    x_pred = [np.append(X, current_composition.composition)[len(current_composition.composition)::]]
    y_pred = reg.predict(x_pred)[0]

    # Setting negative values to zero
    y_pred[y_pred < 0] = 0
    # Normalising
    y_pred = y_pred / sum(y_pred)

    # Picking top n  with highest chances
    highest_indices = y_pred.argsort()[-n::]
    # Used to create output
    zeroes = np.zeros(len(y_pred))
    for index in highest_indices:
        # Creating resulting vector which has zeroes everywhere except one spot
        result = zeroes.copy()
        result[index] = 1

        chance = y_pred[index]

        # Appending the new note to the composition and multiplying the chance of the current composition with the
        # chance of the new note
        new_composition = current_composition.copy()
        new_chance = new_composition.chance * chance

        # Checking if the new chance is bigger than the limit
        if new_chance < limit_chance:
            continue
        else:
            new_composition.add_new(vector_to_note(min_note, result))
            new_composition.chance = new_chance
            if len(new_composition) == prediction_length:
                allcompositions.add_new(new_composition)
                continue
            else:
                prediction_vector_traverse_recurse(X, reg, min_note, new_composition, allcompositions, n,
                                                   limit_chance=limit_chance*global_limit_chance,
                                                   global_limit_chance=global_limit_chance,
                                                   prediction_length=prediction_length)



def vector_to_note(min_note: int, vector) -> int:
    """"
    Function to return a note number from a vectorised note

    :param min_note: minumum note value
    :param vector: the note vector

    :returns key number of the note
    """
    # Quick check, if the first element is a 1 then the note is zero, else the note is
    # simply shifter by the minimum note
    if vector[0]==1:
        return 0
    else:
        return int(np.where(vector==1)[0][0] + min_note - 1)

if __name__ == "__main__":
    print(vector_to_note(23,np.array([0,0,1,0])))