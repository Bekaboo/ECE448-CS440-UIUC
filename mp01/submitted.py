'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import itertools

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    word_count = []
    for words in texts:
        word_count.append([words.count(word0), words.count(word1)])

    max_count = int(np.max(word_count))
    Pjoint = np.zeros((max_count + 1, max_count + 1), np.uint)
    for word_file_count in word_count:
        Pjoint[word_file_count[0], word_file_count[1]] += 1
    Pjoint = Pjoint / len(texts)

    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    index_to_axis = [1, 0]
    return np.array(Pjoint).sum(axis=index_to_axis[index])

def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    return np.divide(Pjoint, Pmarginal[:, np.newaxis])

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    mu (float) - the mean of X
    '''
    return np.arange(len(P)) @ P

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    var (float) - the variance of X
    '''
    return (np.arange(len(P)) - mean_from_distribution(P)) ** 2 @ P

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)

    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    mean0 = mean_from_distribution(P.sum(axis=1))
    mean1 = mean_from_distribution(P.sum(axis=0))

    P_0_mul_1 = np.zeros(P.shape[0] * P.shape[1] + 1)
    for entry in itertools.product(np.arange(P.shape[0]),
                                   np.arange(P.shape[1])):
        P_0_mul_1[entry[0] * entry[1]] += P[entry[0], entry[1]]
    mean_0_mul_1 = mean_from_distribution(P_0_mul_1)

    return mean_0_mul_1 - mean0 * mean1

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0
    for entry in itertools.product(np.arange(P.shape[0]),
                                   np.arange(P.shape[1])):
        expected += P[entry[0], entry[1]] * f(entry[0], entry[1])

    return expected
