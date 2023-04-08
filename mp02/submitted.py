'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import itertools
import copy
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists)
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters)
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    frequency = {}
    for text_class in train:
        frequency[text_class] = Counter()
        for text in train[text_class]:
            for token in text:
                frequency[text_class][token] += 1
    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters)
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    nonstop = copy.deepcopy(frequency)
    for text_class in nonstop:
        for stopword in stopwords:
            if stopword in nonstop[text_class]:
                del nonstop[text_class][stopword]
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts)
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    for text_class in nonstop:
        n_tokens_in_class = sum(nonstop[text_class].values())
        n_word_types_in_class = len(nonstop[text_class])

        nonstop[text_class]['OOV'] = \
            smoothness / (n_tokens_in_class + \
                          smoothness * (1 + n_word_types_in_class))

        for word in nonstop[text_class]:
            n_tokens_of_word_in_class = nonstop[text_class][word]
            nonstop[text_class][word] = \
                (n_tokens_of_word_in_class + smoothness) / \
                (n_tokens_in_class + smoothness * (1 + n_word_types_in_class))

    return nonstop

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts)
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    for text in texts:
        p_pos_given_text = np.log(prior)
        p_neg_given_text = np.log(1 - prior)
        for token in text:
            if token in stopwords:
                continue
            p_pos_given_text += np.log(likelihood['pos'][token]) \
                if token in likelihood['pos'] else np.log(likelihood['pos']['OOV'])
            p_neg_given_text += np.log(likelihood['neg'][token]) \
                if token in likelihood['neg'] else np.log(likelihood['neg']['OOV'])

        hypotheses.append('pos' if p_pos_given_text > p_neg_given_text else 'neg')

    return hypotheses

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters)
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for param_combo in itertools.product(priors, smoothnesses):
        prior = param_combo[0]
        smoothness = param_combo[1]
        likelihood = laplace_smoothing(nonstop, smoothness)
        hypotheses = naive_bayes(texts, likelihood, prior)
        accuracies[priors.index(prior), smoothnesses.index(smoothness)] = \
            np.mean(np.array(hypotheses) == np.array(labels))
    return accuracies
