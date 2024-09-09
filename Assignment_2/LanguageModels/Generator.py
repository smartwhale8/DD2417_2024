import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
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


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
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

    def get_possible_next_words(self, current_word):
        """
        Returns a list of possible next words based on the current word.
        """
        if current_word not in self.index:
            print("\nERROR: \"" + current_word + "\" not found in the corpus. Consider a new start word\n")
            return [] #Return an empty list if the curent word is not in the language model
        
        possible_next_words = []
        probabilities = []

        #check if there are bigram probabilities for the current word
        """by veryifying if its index exist in the bigram_prob dictionary"""
        if self.index[current_word] in self.bigram_prob:
            #Get the dictionary of next word indices and their probabilities
            next_word_indices_probabilities = self.bigram_prob[self.index[current_word]]

            #Convert indices to actual words using the self.word dictionary
            possible_next_words = [self.word[idx] for idx in next_word_indices_probabilities.keys()]
            probabilities = [next_word_indices_probabilities[idx] for idx in next_word_indices_probabilities.keys()]

        return possible_next_words, probabilities

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 
        #Initiailze a sequence with the strarting word
        sequence = [w]

        for _ in range(n):
            # Get a list of possible next words based on the current word
            #sequence[-1] gives the last element of the sequence:
            possible_next_words, probabilities = self.get_possible_next_words(sequence[-1])

            #if there are no possible next words, break the loop
            if not possible_next_words:
                break
            
            #sample the next word from the list of possible next words
            #next_word = random.choice(possible_next_words)

            # Sample the next word from the list of possible next words using the probabilities
            next_word = random.choices(possible_next_words, weights=probabilities, k=1)[0]

            #Add the sampled word to the sequence
            sequence.append(next_word)

        #Print the generated sequence
        print(' '.join(sequence))


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
