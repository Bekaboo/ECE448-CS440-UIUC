"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not
use other non-standard modules (including nltk). Some modules that might be helpful are
already imported for you.
"""

import math
from collections import defaultdict, Counter, namedtuple
from math import log
import numpy as np

# define your epsilon for laplace smoothing here


def baseline(train, test):
    """
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    # Build a lookup table for each word and its tags
    lookup_tbl = {"PHONY": {}}
    for sentence in train:
        for word, tag in sentence:
            # Record the tag in PHONY
            if tag not in lookup_tbl["PHONY"]:
                lookup_tbl["PHONY"][tag] = 1
            else:
                lookup_tbl["PHONY"][tag] += 1
            # Record the tag in the word
            if word not in lookup_tbl:
                lookup_tbl[word] = {tag: 1}
                continue
            # word in lookup_tbl
            if tag not in lookup_tbl[word]:
                lookup_tbl[word][tag] = 1
                continue
            # word, tag in lookup_tbl
            lookup_tbl[word][tag] += 1

    # Tag words in test data
    tagged_test = []
    for sentence in test:
        tagged_sentence = []
        for word in sentence:
            lookup_word = "PHONY" if word not in lookup_tbl else word
            tagged_sentence.append(
                (word, max(lookup_tbl[lookup_word], key=lookup_tbl[lookup_word].get))
            )
        tagged_test.append(tagged_sentence)

    return tagged_test


TrellisNode = namedtuple("TrellisNode", ["tag", "prev_node", "log_prob"])


def viterbi(train, test):
    """
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    # probabilities[tag] = {
    #    gen_word: {probability to generate given words},
    #    next_tag: {probability to go to next tag}
    # }
    probabilities = {}

    # Traverse the training data to calculate the probabilities
    for sentence in train:
        for idx, (word, tag) in enumerate(sentence):
            # Create entry for tag if not exist
            if tag not in probabilities:
                probabilities[tag] = {"gen_word": {}, "next_tag": {}}
            # Record number of occurrence of generated word
            if word not in probabilities[tag]["gen_word"]:
                probabilities[tag]["gen_word"][word] = 0
            probabilities[tag]["gen_word"][word] += 1
            # Record number of occurrence of next tag
            if tag != sentence[-1][1]:
                next_tag = sentence[idx + 1][1]
                if next_tag not in probabilities[tag]["next_tag"]:
                    probabilities[tag]["next_tag"][next_tag] = 0
                probabilities[tag]["next_tag"][next_tag] += 1

    # Calculate the probabilities for each tag in tags, using Laplace smoothing
    k = 1e-5
    for tag in probabilities:
        # Calculate probability of generating words
        total_gen_word = sum(probabilities[tag]["gen_word"].values())
        probabilities[tag]["gen_word"]["PHONY"] = k
        for word in probabilities[tag]["gen_word"]:
            probabilities[tag]["gen_word"][word] = (
                probabilities[tag]["gen_word"][word] + k
            ) / (total_gen_word + k * len(probabilities[tag]["gen_word"]))

        # Calculate probability of going to next tag
        total_next_tag = sum(probabilities[tag]["next_tag"].values())
        probabilities[tag]["next_tag"]["PHONY"] = k
        for next_tag in probabilities[tag]["next_tag"]:
            probabilities[tag]["next_tag"][next_tag] = (
                probabilities[tag]["next_tag"][next_tag] + k
            ) / (total_next_tag + k * len(probabilities[tag]["next_tag"]))
    # By now we have estimated the probabilities of the generated words and
    # the probabilities of possible next tags for each tag

    # Tag each sentence in test data
    for sentence in test:
        if len(sentence) == 0:
            continue
        # Using Viterbi algorithm to tag a sentence the test data
        # trellis = [
        #     [
        #         (word_0_candidate_tag_0, previous_tag_00, current_log_probability_00),
        #         (word_0_candidate_tag_1, previous_tag_01, current_log_probability_01),
        #         ...
        #     ],
        #     [
        #         (word_1_candidate_tag_0, previous_tag_10, current_log_probability_10),
        #         (word_1_candidate_tag_1, previous_tag_11, current_log_probability_11),
        #         ...
        #     ],
        #     ...
        # ]
        trellis = [[TrellisNode("START", "START", 1)]]
        for idx, word in enumerate(sentence[1:], 1):
            trellis.append([])
            for cur_tag in probabilities:
                # For each possible tag at current position, find the previous
                # tag that has the highest probability to go to current tag
                best_prev_node = None
                best_log_prob = -math.inf
                for prev_node in trellis[idx - 1]:
                    next_tag_probs = probabilities[prev_node.tag]["next_tag"]
                    gen_word_probs = probabilities[cur_tag]["gen_word"]
                    # fmt: off
                    cur_log_prob = (
                        prev_node.log_prob \
                        + math.log(next_tag_probs[cur_tag if cur_tag in next_tag_probs else "PHONY"]) \
                        + math.log(gen_word_probs[word if word in gen_word_probs else "PHONY"])
                    )
                    # fmt: on
                    if cur_log_prob > best_log_prob:
                        best_prev_node = prev_node
                        best_log_prob = cur_log_prob
                trellis[idx].append(TrellisNode(cur_tag, best_prev_node, best_log_prob))

        # Find the tag that has the highest probability to go to "END"
        current_trellis_node = max(trellis[-1], key=lambda x: x.log_prob)
        sentence[-1] = (sentence[-1], current_trellis_node.tag)
        for idx, word in reversed(list(enumerate(sentence[:-1]))):
            sentence[idx] = (word, current_trellis_node.prev_node.tag)
            current_trellis_node = current_trellis_node.prev_node

    return test


def _find_hapax(train):
    word_count = {}
    for sentence in train:
        for word, tag in sentence:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
    return [word for word, count in word_count.items() if count == 1]


def viterbi_ec(train, test):
    """
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    # hapax_words = list of hapax words
    # hapax_probabilities[tag] = probabilities of each tag given a hapax word
    hapax_words = _find_hapax(train)
    hapax_probabilities = {}

    # probabilities[tag] = {
    #    gen_word: {probability to generate given words},
    #    next_tag: {probability to go to next tag}
    # }
    probabilities = {}

    # Traverse the training data to calculate the probabilities
    for sentence in train:
        for idx, (word, tag) in enumerate(sentence):
            # Record the tag if it is a hapax word
            if word in hapax_words:
                if tag not in hapax_probabilities:
                    hapax_probabilities[tag] = 0
                hapax_probabilities[tag] += 1

            # Create entry for tag if not exist
            if tag not in probabilities:
                probabilities[tag] = {"gen_word": {}, "next_tag": {}}
            # Record number of occurrence of generated word
            if word not in probabilities[tag]["gen_word"]:
                probabilities[tag]["gen_word"][word] = 0
            probabilities[tag]["gen_word"][word] += 1
            # Record number of occurrence of next tag
            if tag != sentence[-1][1]:
                next_tag = sentence[idx + 1][1]
                if next_tag not in probabilities[tag]["next_tag"]:
                    probabilities[tag]["next_tag"][next_tag] = 0
                probabilities[tag]["next_tag"][next_tag] += 1

    # Calculate the probabilities for hapax words, using Laplace smoothing
    k_hapax = 1e-5
    total_hapax_tag_count = sum(hapax_probabilities.values())
    hapax_probabilities["PHONY"] = k_hapax
    for tag in hapax_probabilities:
        hapax_probabilities[tag] = (hapax_probabilities[tag] + k_hapax) / (
            total_hapax_tag_count + k_hapax * len(hapax_probabilities)
        )

    # Calculate the probabilities for each tag in tags, using Laplace smoothing
    k = 1e-5
    for tag in probabilities:
        # Calculate probability of generating words
        k_scaled = (
            k * hapax_probabilities[tag if tag in hapax_probabilities else "PHONY"]
        )
        total_gen_word = sum(probabilities[tag]["gen_word"].values())
        probabilities[tag]["gen_word"]["PHONY"] = k_scaled
        for word in probabilities[tag]["gen_word"]:
            probabilities[tag]["gen_word"][word] = (
                probabilities[tag]["gen_word"][word] + k_scaled
            ) / (total_gen_word + k_scaled * len(probabilities[tag]["gen_word"]))

        # Calculate probability of going to next tag
        total_next_tag = sum(probabilities[tag]["next_tag"].values())
        probabilities[tag]["next_tag"]["PHONY"] = k
        for next_tag in probabilities[tag]["next_tag"]:
            probabilities[tag]["next_tag"][next_tag] = (
                probabilities[tag]["next_tag"][next_tag] + k
            ) / (total_next_tag + k * len(probabilities[tag]["next_tag"]))

    # Tag each sentence in test data
    for sentence in test:
        if len(sentence) == 0:
            continue
        # Using Viterbi algorithm to tag a sentence the test data
        # trellis = [
        #     [
        #         (word_0_candidate_tag_0, previous_tag_00, current_log_probability_00),
        #         (word_0_candidate_tag_1, previous_tag_01, current_log_probability_01),
        #         ...
        #     ],
        #     [
        #         (word_1_candidate_tag_0, previous_tag_10, current_log_probability_10),
        #         (word_1_candidate_tag_1, previous_tag_11, current_log_probability_11),
        #         ...
        #     ],
        #     ...
        # ]
        trellis = [[TrellisNode("START", "START", 1)]]
        for idx, word in enumerate(sentence[1:], 1):
            trellis.append([])
            for cur_tag in probabilities:
                # For each possible tag at current position, find the previous
                # tag that has the highest probability to go to current tag
                best_prev_node = None
                best_log_prob = -math.inf
                for prev_node in trellis[idx - 1]:
                    next_tag_probs = probabilities[prev_node.tag]["next_tag"]
                    gen_word_probs = probabilities[cur_tag]["gen_word"]
                    # fmt: off
                    cur_log_prob = (
                        prev_node.log_prob \
                        + math.log(next_tag_probs[cur_tag if cur_tag in next_tag_probs else "PHONY"]) \
                        + math.log(gen_word_probs[word if word in gen_word_probs else "PHONY"])
                    )
                    # fmt: on
                    if cur_log_prob > best_log_prob:
                        best_prev_node = prev_node
                        best_log_prob = cur_log_prob
                trellis[idx].append(TrellisNode(cur_tag, best_prev_node, best_log_prob))

        # Find the tag that has the highest probability to go to "END"
        current_trellis_node = max(trellis[-1], key=lambda x: x.log_prob)
        sentence[-1] = (sentence[-1], current_trellis_node.tag)
        for idx, word in reversed(list(enumerate(sentence[:-1]))):
            sentence[idx] = (word, current_trellis_node.prev_node.tag)
            current_trellis_node = current_trellis_node.prev_node

    return test
