#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import faulthandler
faulthandler.enable()
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs

"""
This file is part of the computer assignments for the course DD2417 Language Engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file f.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = text_file.read().encode('utf-8').decode().lower()
        try :
            self.tokens = nltk.word_tokenize(text) 
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE - START        
        self.total_words += 1

        #update the unigram count
        """if token does not exist in the dictionary unigram_count, then we are encountering the token for the first time
            hence, must update:
                unique_words count 
                word[new token index] and 
                index[token]
        """
        if token not in self.unigram_count:
            #it means we haven't yet seen this token
            #a new unique word added:
            self.unique_words += 1
            new_index = len(self.index)
            self.word[new_index] = token
            self.index[token] = new_index
        
        #increment the count of the unigram: 
        #will also create a new entry if token doesn't yet exist  (defaultdict)
        self.unigram_count[token] += 1

        """
        ensure that last token processed is not the second-to-last token in the corpus
        last_index holds the index of the last token processed in the corpus.
        
        self.last_index + 2: Adding 2 to self.last_index gives the index of the token 
        that occurs two positions after the last token processed. This represents the 
        next token in the corpus after the one currently being processed.

        self.bigram_count[token][self.last_index + 2] accesses the bigram count for 
        the pair formed by the current token and the token two positions ahead (i.e., the next token) 
        in the list of tokens.
        """
        if self.last_index != len(self.tokens) -2:
            self.bigram_count[token][self.tokens[self.last_index + 2]] += 1
        
        self.last_index += 1
        # YOUR CODE HERE - END

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        rows_to_print = []

        # YOUR CODE HERE - START
        rows_to_print.append(str(self.unique_words)+ " " + str(self.total_words))
        for ident in self.word.keys():
            rows_to_print.append(str(ident) + " " + self.word[ident] + " " + str(self.unigram_count[self.word[ident]]))
        
        for word in self.index.keys():
            sum = math.fsum(self.bigram_count[word].values())
            for pairing in self.bigram_count[word]:
                rows_to_print.append(str(self.index[word]) + " " + str(self.index[pairing]) + " " + '{:01.15f}'.format(math.log(self.bigram_count[word][pairing]/sum)))
        
        rows_to_print.append("-1")
        # YOUR CODE HERE - END
        return rows_to_print

    def __init__(self):
        """
        Constructor. Processes the file f and builds a language model
        from it.

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)


if __name__ == "__main__":
    main()
