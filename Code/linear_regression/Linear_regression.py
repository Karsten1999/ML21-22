import numpy as np 

def Get_weight_estimate(n, voice):
    """
    Function which estimates the weights for linear regression.
    
    Steps followed are from section 3.1 of the lecture notes (eq. 19)
    
    The input vector x_i in this case are taken as arrays of length n, 
    with the entries being the previous n tones
    The output vector y_i is the 'next tone', which comes after the n 
    tones stored ini x_i
    
    The training data taken is the whole set 'voice', where we loop over 
    all the possible tones, thus N = len(voice) - n 
    """
    
    # Initialize array for X and y
    X = np.array(np.ones(n+1).transpose(0)) # Initialize X with ones for 
                                            # correct dimensions
    y = []
    
    # Split the voice in N training points with x having j = n + 1 dimensions.
    # (It has one bias data point)
    for i in range(n, len(voice)-1):       # Maximum training points
        x_i = np.append(voice[i-n:i], [1]) # Add training points and bias point
        X = np.c_[X, x_i.transpose()]      # Format into X
        y.append(voice[i+1])               # Append tone y to array
    
    # Remove the initialised row of ones
    X = X[:, 1:]

    # Compute W' = (XX')^-1 Xy
    W = np.linalg.inv(X.dot(X.transpose())).dot(X.dot(y))    
    
    return W.transpose()


def find_nearest_tone(array,value):
    """
    Function which finds the closest possible tone to the return value
    of the linear regression function.
    """
    idx = (np.abs(array-value)).argmin()
    return idx


def Get_next_tone(n, voice, W, possible_tones):
        """
        Function which finds the next tone, following the linear regression
        method.
        """
        # get last n notes + bias value of 1
        x_i = np.append(voice[-n:], 1).transpose()

        ##################################################################
        # Some selection mechanism should exist here to not only get the #
        # most likely candidate                                          #
        ##################################################################
        
        linear_regression_value = W.dot(x_i)

        new_tone = find_nearest_tone(possible_tones, linear_regression_value)

        return new_tone


def Estimate_linear_regression(voice, n, N):
    """
    Function which finishes Bach's fugue for a single voice, using linear 
    regression.

    n is the value for the number of previous tones to use for estimating the 
    next one
    N is the number of tones which should be generated.    
    """
    # Get the mean of the voices as normalisation factor
    norm_factor = np.mean(voice[voice!=0])
    # Normalize the voice so it is centered around 0
    voice[voice!=0] -= norm_factor
    
    # Get the set of all possible tones
    possible_tones = list(set(voice))
    
    # Get the weight
    W = Get_weight_estimate(n, voice)

    # Store voice in a new array
    voice_new = voice

    # Estimate the upcoming N_ tones
    for i in range(N):
        voice_new = np.append(voice_new,
                      Get_next_tone(n, voice_new, W, possible_tones))
    
    # reformat the normalised new voice into original format
    voice_new[voice_new!=0] += norm_factor
    
    # Return the finished piece
    return voice_new
