#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: True if the entire file could be processed, False otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                print("file " + filename + " has been opened: \nUnique word count: " + str(self.unique_words) + ", total word count: " + str(self.total_words))

                #Read unigram counts and populate data structures
                for _ in range(self.unique_words):
                    idx, token, count = f.readline().strip().split()
                    idx = int(idx)
                    count = int(count)
                    #print(f"idx: {idx}, token: {token}, count: {count}")
                    self.word[idx] = token
                    self.index[token] = idx
                    self.unigram_count[token] = count

                #Read bigram probabilities and populate data structures
                tokens = f.readline().strip().split()
                while int(tokens[0]) >= 0:
                    first_word_idx, second_word_idx = map(int, tokens[:2])
                    log_prob = float(tokens[2])
                    self.bigram_prob[first_word_idx][second_word_idx] =  math.exp(log_prob) #we convert back from log-prob to original prob
                    #print(f"first_word_idx: {first_word_idx}, second_word_idx: {second_word_idx}, log_prob: {log_prob}")
                    tokens = f.readline().strip().split()
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False


    def compute_entropy_cumulatively(self, word):
        iter_cross_entropy = -1 / len(self.tokens) #we will accumulate the cross-entropy for the current word

        if word not in self.word.values():
            #Handle missing word in the model -- must assing a very small probability - represents an unknown word chance
            bigram_prob = self.lambda3
        elif self.last_index == -1 or self.tokens[self.last_index] not in self.word.values():
            # Handle the case where there's no previous word or the previous word is missing (not in model)
            prev_unigram_prob = self.unigram_count.get(word, 0) / (self.total_words or 1)  # Handle division by zero
            bigram_prob = self.lambda2 * prev_unigram_prob + self.lambda3

        else:
            # Calculate bigram probability using linear interpolation
            # the regular case where both the current and the previous words are known and in the model
            prev_word = self.tokens[self.last_index]
            prev_word_index = self.index.get(prev_word, -1)
            current_word_index = self.index.get(word, -1)        

            bigram_prob = (
                self.lambda1 * self.bigram_prob.get(prev_word_index, {}).get(current_word_index, 0) + 
                self.lambda2 * self.unigram_count.get(word, 0) / max(1, self.total_words) + 
                self.lambda3
            )

        #update the cumulative log probability
        iter_cross_entropy *= math.log(bigram_prob)  #prob are typically small, less than 1, the log of prob is negative, making the product positive

        self.logProb += iter_cross_entropy
        self.test_words_processed += 1
        self.last_index += 1

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) 
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()
