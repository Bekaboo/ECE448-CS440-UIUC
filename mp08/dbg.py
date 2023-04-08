import utils
import submitted
import importlib
import time

train_set = utils.load_dataset("data/brown-training.txt")
dev_set = utils.load_dataset("data/brown-test.txt")
tag_fn = submitted.viterbi_ec

###############################################################################
importlib.reload(submitted)
train_set = utils.load_dataset("data/brown-training.txt")
dev_set = utils.load_dataset("data/brown-test.txt")
test_set = dev_set[0:1]
print("======== test_set ========\n{}".format(test_set))
start_time = time.time()
predicted = tag_fn(train_set, utils.strip_tags(test_set))
print("======== predicted =======\n{}".format(predicted))
time_spend = time.time() - start_time
accuracy, _, _ = utils.evaluate_accuracies(predicted, test_set)
(
    multi_tag_accuracy,
    unseen_words_accuracy,
) = utils.specialword_accuracies(train_set, predicted, test_set)

print("time spent: {0:.4f} sec".format(time_spend))
print("accuracy: {0:.4f}".format(accuracy))
print("multi-tag accuracy: {0:.4f}".format(multi_tag_accuracy))
print("unseen word accuracy: {0:.4f}".format(unseen_words_accuracy))
