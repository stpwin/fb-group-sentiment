# -*- coding: utf-8 -*-
import nltk
import re
import codecs
import pickle
import os.path
from pythainlp.tokenize import word_tokenize
from prettytable import PrettyTable


# tknzr = nltk.tokenize.TweetTokenizer()

word_features = []


def build_vocabulary(training_data_set):
    all_words = []
    for (words, sentiment) in training_data_set:
        all_words.extend(words)

    word_list = nltk.FreqDist(all_words)
    # word_features = [word[0] for word in word_list.most_common()]
    # print("word_features most_common", word_features)
    # print("")
    # print("word_features keys", word_features)
    return [word[0] for word in word_list.most_common()]  # word_list.keys()


def extract_features(document: list):  # { 'contains(word)': True/False }, ...
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f"contains({word})"] = (word in document_words)
    # print("\nfeature", features)
    return features


def read_corpus(filename, tag):
    _read = codecs.open(filename, 'r', "utf-8")
    reviews = []
    for each_review in _read:
        each_review = ''.join(word_tokenize(each_review))
        if each_review.endswith('\n'):
            each_review = each_review[:-1]
        if each_review.endswith('\r'):
            each_review = each_review[:-1]
        if not each_review == '':
            reviews.append([each_review, tag])
    return reviews


def prepare_data():  # [('word', 'sentiment'), ...]
    _dir = "test"  # test simple old
    pos_reviews = read_corpus(f'corpus/{_dir}/pos.txt', 'pos')
    neg_reviews = read_corpus(f'corpus/{_dir}/neg.txt', 'neg')
    swear_reviews = read_corpus(f'corpus/{_dir}/swear.txt', 'swear')
    # q_reviews = read_corpus(f'corpus/{dir}/q.txt', 'q')

    training_data_set = []
    for (review, sentiment) in swear_reviews + neg_reviews:
        reviews_filtered = [w.lower() for w in word_tokenize(review)]
        training_data_set.append((reviews_filtered, sentiment))

    return training_data_set


def train(preprocessed_training_set):
    global word_features
    # words = get_words_in_reviews(preprocessedTrainingSet)
    word_features = build_vocabulary(preprocessed_training_set)

    train_set = nltk.classify.apply_features(
        extract_features, preprocessed_training_set)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return classifier


def KFoldAccuracy(all_reviews):
    global word_features
    from sklearn.model_selection import KFold
    import numpy as np
    import collections

    k_fold = KFold(n_splits=10, random_state=1992, shuffle=True)
    featuresets = np.array(all_reviews)
    accuracy_scores = []

    for train_set, test_set in k_fold.split(featuresets):
        word_features = get_word_features(
            get_words_in_reviews(featuresets[train_set].tolist()))
        train_features = nltk.classify.apply_features(
            extract_features, featuresets[train_set].tolist())
        test_features = nltk.classify.apply_features(
            extract_features, featuresets[test_set].tolist())
        classifier = nltk.NaiveBayesClassifier.train(train_features)
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (feats, label) in enumerate(test_features):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)

        accuracy_scores.append(
            nltk.classify.util.accuracy(classifier, test_features))

        # f1_pos = nltk.f_measure(refsets['pos'], testsets['pos'])
        f1_neg = nltk.f_measure(refsets['neg'], testsets['neg'])
        f1_swear = nltk.f_measure(refsets['swear'], testsets['swear'])

        # pre_pos = nltk.precision(refsets['pos'], testsets['pos'])
        pre_neg = nltk.precision(refsets['neg'], testsets['neg'])
        pre_swear = nltk.precision(refsets['swear'], testsets['swear'])

        # re_pos = nltk.recall(refsets['pos'], testsets['pos'])
        re_neg = nltk.recall(refsets['neg'], testsets['neg'])
        re_swear = nltk.recall(refsets['swear'], testsets['swear'])

        print(f'train: {len(train_set)} test: {len(test_set)}')
        print('=================== Results ===================')
        print(f'Accuracy {accuracy_scores[-1]:f}')
        print('            Negative     Swear')
        print(f'F1         {f1_neg:f}     {f1_swear:f}]')
        print(f'Precision  {pre_neg:f}     {pre_swear:f}]')
        print(f'Recall     {re_neg:f}     {re_swear:f}]')
        print('===============================================\n')


def accuracy(all_reviews):
    import numpy as np
    featuresets = np.array(all_reviews)

    train_set, test_set = featuresets[100:], featuresets[:100]

    train_set = nltk.classify.apply_features(extract_features, all_reviews)

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))


def get_words_in_reviews(all_reviews):
    all_words = []
    for (words, sentiment) in all_reviews:
        all_words.extend(words)
    return all_words


def get_word_features(list_of_words: list):
    wordlist = nltk.FreqDist(list_of_words)
    #word_features = [word[0] for word in wordlist.most_common()]
    # return word_features
    return wordlist.keys()


if __name__ == "__main__":
    # if (os.path.exists("my_classifier.pickle")):

    #     data = prepare_data()
    #     words = get_words_in_reviews(data)
    #     word_features = buildVocabulary(words)

    #     import pickle
    #     f = open('my_classifier.pickle', 'rb')
    #     classifier = pickle.load(f)
    #     f.close()
    #     print("load lasted classifier")
    # else:
    preprocessedTrainingSet = prepare_data()
    classifier = train(preprocessedTrainingSet)
    KFoldAccuracy(preprocessedTrainingSet)

    # f = open("my_classifier.pickle", "wb")
    # pickle.dump(classifier, f)
    # f.close()
    # print("saved classifier")

    # read_in = input('Enter >>> ')
    # while read_in != 'q':
    #     tokenized = word_tokenize(read_in, keep_whitespace=False)
    #     extracted_features = extract_features(tokenized)
    #     result = classifier.classify(extracted_features)
    #     entities = [k[9:-1]
    #                 for k, v in extracted_features.items() if v == True]
    #     print("Tokenized:", "|".join(tokenized))
    #     print("Entities:", "|".join(entities))
    #     print("")
    #     # print("Extracted features:", extracted_features)
    #     if len(entities) > 0:
    #         dist = classifier.prob_classify(extracted_features)
    #         # print(dist.samples())
    #         # for label in dist.samples():
    #         # print("%s: %f" % (label, dist.prob(label)))

    #         # samples = dist.samples()

    #         t = PrettyTable(['Sentiment', 'Probability'])
    #         t.add_row([f'{"*pos*" if result == "pos" else "pos"}',
    #                    f"{dist.prob('pos'):f}"])
    #         t.add_row([f'{"*swear*" if result == "swear" else "swear"}',
    #                    f"{dist.prob('swear'):f}"])
    #         t.add_row([f'{"*neg*" if result == "neg" else "neg"}',
    #                    f"{dist.prob('neg'):f}"])
    #         print(t)
    #         print("")
    #     else:
    #         print("No match")
    #         print("")

    #     read_in = input('Enter >>> ')
