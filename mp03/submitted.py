'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np
import itertools

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''

    neighbors = np.zeros((k, np.shape(train_images[0])[0]))
    smallest_distances = np.full(k, np.inf)
    labels = np.full(k, False)

    for img_idx in range(len(train_images)):
        train_image = train_images[img_idx]
        train_label = train_labels[img_idx]
        distance = np.linalg.norm(image - train_image)
        if np.any(distance < smallest_distances) or \
                (np.any(distance == smallest_distances) and not train_label):
            idx = np.argmax(smallest_distances)
            smallest_distances[idx] = distance
            neighbors[idx] = train_image
            labels[idx] = train_label

    return neighbors, labels



def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''

    hypotheses = np.zeros(len(dev_images), dtype='int')
    scores = np.zeros(len(dev_images), dtype='int')

    for image_idx in range(len(dev_images)):
        dev_image = dev_images[image_idx]
        _, labels = k_nearest_neighbors(dev_image, train_images, train_labels, k)
        unique_labels, counts = np.unique(labels, return_counts=True)
        hypotheses[image_idx] = unique_labels[np.argmax(counts)]
        scores[image_idx] = counts[np.argmax(counts)]

    return hypotheses, scores


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    unique_hypotheses = np.unique(hypotheses).astype(int)
    unique_references = np.unique(references).astype(int)
    confusions = np.zeros((len(unique_hypotheses), len(unique_references)))

    for hyp, ref in itertools.product(unique_hypotheses, unique_references):
        confusions[ref][hyp] = np.sum((hyp == hypotheses) & (ref == references))

    accuracy = np.sum(hypotheses == references) / len(references)
    precision = confusions[1][1] / (confusions[1][1] + confusions[0][1])
    recall = confusions[1][1] / (confusions[1][1] + confusions[1][0])
    f1 = 2 * precision * recall / (precision + recall)

    return confusions, accuracy, f1
