import time
from utils import *
import numpy as np
import math


def calculate_train_vocab_prob(train, vocab, alpha):
    # Create Matrix
    counter = []
    for i in range(len(train)):
        counter_per_instance = []
        for j in vocab:
            counter_per_instance.append(train[i].count(j))
        counter.append(counter_per_instance)
    matrix = np.array(counter)
    # Calculate probability for each vocabulary
    vocab_prob = {}
    vocab_occurance = np.sum(matrix, axis=0)
    total_no_of_words = np.sum(matrix)
    for i in range(len(vocab)):
        vocab_prob[vocab[i]] = (vocab_occurance[i] + alpha) / (total_no_of_words + (len(vocab) * alpha))
    laplace = alpha / (total_no_of_words + (len(vocab) * alpha))
    return vocab_prob, laplace


def calculate_posterior_prob(test, train_doc_prob, train_vocab_prob, laplace):
    test_prob = []
    for test_instance in test:
        test_vocab = np.array(list(set(test_instance)))
        test_instance_prob = train_doc_prob
        for vocab in test_vocab:
            if vocab in train_vocab_prob.keys():
                test_instance_prob = test_instance_prob * train_vocab_prob[vocab]
            else:
                test_instance_prob = test_instance_prob * laplace
        test_prob.append(test_instance_prob)
    return test_prob


def calculate_log_prob(test, train_doc_prob, train_vocab_prob, laplace):
    test_prob = []
    for test_instance in test:
        test_vocab = np.array(list(set(test_instance)))
        test_instance_prob = math.log10(train_doc_prob)
        for vocab in test_vocab:
            if vocab in train_vocab_prob.keys():
                test_instance_prob = test_instance_prob + math.log10(train_vocab_prob[vocab])
            else:
                test_instance_prob = test_instance_prob + math.log10(laplace)
        test_prob.append(test_instance_prob)
    return test_prob


def calculate_test_prob(test, pos_doc_prob, neg_doc_prob, pos_vocab_prob, neg_vocab_prob, laplace, use_log):
    if use_log:
        pos_prob = calculate_log_prob(test, pos_doc_prob, pos_vocab_prob, laplace)
        neg_prob = calculate_log_prob(test, neg_doc_prob, neg_vocab_prob, laplace)
    else:
        pos_prob = calculate_posterior_prob(test, pos_doc_prob, pos_vocab_prob, laplace)
        neg_prob = calculate_posterior_prob(test, neg_doc_prob, neg_vocab_prob, laplace)
    prob = []
    for i in range(len(test)):
        if pos_prob[i] > neg_prob[i]:
            prob.append("positive")
        else:
            prob.append("negative")

    return prob


def build_confusion_matrix(pos_test_prob, neg_test_prob):
    tp = pos_test_prob.count("positive")
    fn = pos_test_prob.count("negative")
    fp = neg_test_prob.count("positive")
    tn = neg_test_prob.count("negative")
    return tp, fn, fp, tn


def calculate_performance(pos_test_prob, neg_test_prob, alpha):
    tp, fn, fp, tn = build_confusion_matrix(pos_test_prob, neg_test_prob)
    n = len(pos_test_prob) + len(neg_test_prob)
    confusion_matrix = [[tp, fn], [fp, tn]]
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Alpha: ", alpha)
    print("Confusion Matrix: ", confusion_matrix)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)


def naive_bayes():
    percentage_positive_instances_train = 0.5
    percentage_negative_instances_train = 0.5

    percentage_positive_instances_test = 1
    percentage_negative_instances_test = 1

    alpha = 10.0

    start = time.time()
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    with open('vocab.txt', 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)
    print("Vocabulary (training set):", len(vocab))

    vocab = list(vocab)
    pos_doc_prob = len(pos_train) / (len(pos_train) + len(neg_train))
    neg_doc_prob = len(neg_train) / (len(pos_train) + len(neg_train))

    pos_vocab_prob, laplace = calculate_train_vocab_prob(pos_train, vocab, alpha)
    neg_vocab_prob, laplace = calculate_train_vocab_prob(neg_train, vocab, alpha)

    pos_test_prob = calculate_test_prob(pos_test, pos_doc_prob, neg_doc_prob, pos_vocab_prob, neg_vocab_prob,
                                        laplace, True)
    neg_test_prob = calculate_test_prob(neg_test, pos_doc_prob, neg_doc_prob, pos_vocab_prob, neg_vocab_prob,
                                        laplace, True)
    calculate_performance(pos_test_prob, neg_test_prob, alpha)

    end = time.time()
    print("Time Taken: ", end - start)


if __name__ == "__main__":
    naive_bayes()
