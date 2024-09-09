# Assignment 2

## LanguageModels

This section explores the bigram langauge model.

**bigram model** is a simple and efficient way to capture word-to-word dependencies, making it a foundational concept in statistical language processing.

A **bigram model** is a type of statistical language model that predicts the probability of a word given the preceding word in a sequence. It is based on the **Markov assumption**, which simplifies language modeling by assuming that the probability of a word depends only on the previous word, rather than on the entire preceding sequence.

### **Key Concepts:**

1. **N-gram Language Models:**
   - N-gram models predict the probability of a word based on the last **N-1** words. A bigram model is a special case where **N=2**, meaning that the model looks at only one preceding word to make predictions.

2. **Bigram Probability:**
   - The probability of a word \( w_i \) given the previous word \( w_{i-1} \) is expressed as:
     \[
     P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}
     \]
   - This is the conditional probability of a word occurring given that the previous word has already occurred.

3. **Training:**
   - A bigram model is trained by calculating the frequencies of word pairs (bigrams) in a given corpus.
   - From these frequencies, the model computes the conditional probabilities for each bigram.

4. **Usage:**
   - **Text Generation:** The bigram model can be used to generate text by repeatedly sampling the next word based on the previous word's probability distribution.
   - **Speech Recognition & Machine Translation:** Bigram models are used in tasks that involve predicting the next word or phrase based on context.
   - **Language Modeling:** They provide a simple way to estimate the likelihood of a sequence of words in a language, which is useful in applications like spelling correction or autocomplete.

5. **Limitations:**
   - **Limited Context:** The bigram model only considers the immediate previous word, which makes it less accurate for long-range dependencies and complex linguistic patterns.
   - **Sparsity:** Many word pairs may not occur in the training data, leading to sparsity issues. Techniques like smoothing are used to address this.

### **Example:**
In the sentence "the cat sat on the mat," the bigrams are:
- ("the", "cat")
- ("cat", "sat")
- ("sat", "on")
- ("on", "the")
- ("the", "mat")

The bigram model calculates the probability of each word in the sentence given the previous word and uses these probabilities to model the sentence.

As part of the project, we build a bigram language model from a given corpus. The file ```BigramTrainer.py``` reads the corpus, processes the tokens, and then calculates the unigram and bigram counts. The results are either printed or saved to a file.

### **Summary of Functions:**

1. **`__init__()` (BigramTrainer class constructor):**
   - Initializes various attributes to track word-to-identifier mappings, unigram and bigram counts, and the total/unique word counts.
   - Uses `defaultdict` to handle missing keys for unigrams and bigrams efficiently.

2. **`process_files(f)` (BigramTrainer class):**
   - Reads the file `f`, converts it to lowercase, and tokenizes the text using the NLTK library.
   - Calls `process_token()` on each token to update unigram and bigram counts.

3. **`process_token(token)` (BigramTrainer class):**
   - Updates unigram and bigram counts for the given token.
   - Keeps track of unique words and total words processed.
   - Updates bigram counts for word pairs if not at the end of the corpus.

4. **`stats()` (BigramTrainer class):**
   - Generates a list of rows summarizing the unigram and bigram statistics.
   - For each word, it appends the word's identifier, frequency, and bigram probabilities (log probabilities) to the output.
   - Returns the formatted statistics list.

5. **`main()` function:**
   - Parses command line arguments to specify the input file and optional output file.
   - Instantiates the `BigramTrainer` class and processes the input file.
   - Retrieves the language model statistics via `stats()` and writes them to the destination file or prints them to the console.

The output of this code is a formatted representation of the bigram language model created from the input text file. This output includes information about the total number of unique words, total word counts, unigram counts, and bigram probabilities.

### **Output Breakdown:**

1. **First Line:**
   - Contains two numbers separated by a space:
     - The first number is the **total number of unique words** (vocabulary size) in the corpus.
     - The second number is the **total number of words** processed (total token count).

   **Example:**
500 10000
- This indicates that there are 500 unique words, and the total word count is 10,000.

2. **Unigram Information:**
- For each word in the vocabulary, a line is printed with the following format:
  - **Word ID**: The unique integer identifier assigned to the word.
  - **Word**: The actual word itself.
  - **Unigram Count**: The frequency of the word in the corpus.

**Example:**
```
0 the 500 
1 cat 50 
2 sat 25
```

- This indicates that "the" (ID: 0) appears 500 times, "cat" (ID: 1) appears 50 times, and "sat" (ID: 2) appears 25 times.

3. **Bigram Information:**
- For each bigram (a pair of consecutive words), the following format is printed:
  - **First Word ID**: The ID of the first word in the bigram.
  - **Second Word ID**: The ID of the second word in the bigram.
  - **Log Probability**: The logarithm of the probability of the second word following the first word, computed as:
    \[
    \log\left(\frac{\text{Bigram Count (first word, second word)}}{\sum \text{Bigram Count (first word, *)}}\right)
    \]
  - This represents the probability of transitioning from the first word to the second word in the corpus.

**Example:**
```
0 1 -1.2039728043259361 
1 2 -2.3025850929940459
```
 
- This indicates that the bigram formed by "the" (ID: 0) followed by "cat" (ID: 1) has a log probability of -1.2039, and the bigram "cat" (ID: 1) followed by "sat" (ID: 2) has a log probability of -2.3025.

4. **End Marker:**
- The final line of the output is simply:
  ```
  -1
  ```
- This serves as a marker indicating the end of the output.

### **Example Output:**
If the corpus contains sentences like "the cat sat," the output could look something like:
```
3 3 
0 the 1
1 cat 1
2 sat 1
0 1 -0.6931471805599453
1 2 -0.6931471805599453
-1
```
 
In this case, there are 3 unique words, 3 total tokens, and two bigrams with equal probabilities.

```
# Creating various models from their respective corpus

# To create guardian_model from guardian_training data
$  python BigramTrainer.py -f .\data\guardian_training.txt -d guardian_model.txt

# To create austen_model from austen_training data
$ python .\BigramTrainer.py -f .\data\austen_training.txt -d austen_model.txt

# To create kafka_model from kafka data
$ python .\BigramTrainer.py -f .\data\kafka.txt -d .\kafka_model.txt

```

## Text Generation

The file ```Generator.py``` contains the code to generate words from a bigram language model. It reads a pre-trained language model from a file and generates sequences of words by sampling from the bigram probabilities. The output is a sequence of words based on the specified starting word and the number of words to generate.

### **Summary of Functions:**

1. **`__init__()` (Generator class constructor):**
   - Initializes data structures to store word-to-identifier mappings, unigram counts, bigram probabilities, and other statistics.
   - Sets parameters for smoothing with lambda values and tracks the number of test words processed.

2. **`read_model(filename)` (Generator class):**
   - Reads a language model from a file.
   - Populates the data structures with unigram counts and bigram probabilities.
   - Converts log-probabilities back to original probabilities for further use.
   - Returns `True` if the file is successfully read; otherwise, prints an error message and returns `False`.

3. **`get_possible_next_words(current_word)` (Generator class):**
   - Given a current word, retrieves a list of possible next words along with their corresponding probabilities.
   - If the current word is not found in the model, returns an empty list and prints an error message.
   - This function checks if bigram probabilities exist for the current word and converts word indices to actual words.

4. **`generate(w, n)` (Generator class):**
   - Generates and prints `n` words starting with the word `w`.
   - Uses bigram probabilities to sample the next word based on the current word.
   - The generated sequence is printed after the loop terminates, either when `n` words are generated or no further words can be generated.

5. **`main()` function:**
   - Parses command line arguments to specify the language model file, the starting word, and the number of words to generate.
   - Instantiates the `Generator` class, reads the language model, and generates a sequence of words based on the provided input.


### **Example Output:**

We use the bigram language model we create earlier for the kafka corups.
```
usage: Generator.py [-h] --file FILE --start START [--number_of_words NUMBER_OF_WORDS]

BigramTester

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  file with language model
  --start START, -s START
                        starting word
  --number_of_words NUMBER_OF_WORDS, -n NUMBER_OF_WORDS

# Generating a 40 word sequence using kafka model
$ python3 Generator.py -f kafka_model.txt -s "the" -n 40
file kafka_model.txt has been opened:
Unique word count: 2605, total word count: 24944
the carpet like we would have the crack in a misfortune had lain in the hall that was saying . they had become a minor discourtesy , so gregor realised it would now , that nothing to the middle of you

# Generating a 20 word sequence using guardian model
$ python .\Generator.py -f .\guardian_model.txt -s "hope" -n 20
file .\guardian_model.txt has been opened:
Unique word count: 172109, total word count: 8910277
hope . this kind of failure of the nla member of bbc trust organic food to make them . related :
```

As a next example, we create a new model from the corpus 'Pride and Prejudice' by Jane Austen, which is available in the file data\austen_training.txt

```
# Create the model
$ python .\BigramTrainer.py -f .\data\austen_training.txt -d austen_model.txt

# Generate a sequence of 20 words using austen model starting with "the"
$ python .\Generator.py -f .\austen_model.txt -s "the" -n 20
file .\austen_model.txt has been opened:
Unique word count: 6422, total word count: 130254
the evening engagements at longbourn again , and february pass away . i entreat him herself upon it is never have
--------------------------------------------------
# Generate a sequence of 20 words using austen model starting with "hope"
$ python .\Generator.py -f .\austen_model.txt -s "hope" -n 20
file .\austen_model.txt has been opened:
Unique word count: 6422, total word count: 130254
hope , if i do not unreasonably have put up some herself not so _very_ easy until the other subject which
--------------------------------------------------
# Generate a sequence of 20 words using guardian model starting with "hope"
python .\Generator.py -f .\guardian_model.txt -s "hope" -n 20
file .\guardian_model.txt has been opened:
Unique word count: 172109, total word count: 8910277
hope . this kind of failure of the nla member of bbc trust organic food to make them . related :
--------------------------------------------------
# Generate a sequence of 20 words using guardian model starting with "news"
python .\Generator.py -f .\guardian_model.txt -s "news" -n 20
file .\guardian_model.txt has been opened:
Unique word count: 172109, total word count: 8910277
news corp columnist , especially to the outside - buy one of national economy was “ every free following lesley land
--------------------------------------------------

```

## Evaluating n-gram models

The `BigramTester.py` defined a class that reads a bigram language model and evaluates the entropy of a test corpus. The entropy measures how well the language model predicts the test data. The script calculates the cumulative entropy by processing each word in the test corpus, comparing it against the probabilities from the bigram model.

### Function Summaries:

1. **`__init__(self)`**:
   - Initializes the `BigramTester` class by setting up the necessary data structures such as mappings of words to identifiers, unigram counts, bigram probabilities, and other variables used in computing entropy. It also sets the lambda values for interpolation of probabilities for unknown words, unigrams, and bigrams.

2. **`read_model(self, filename)`**:
   - Reads the language model from a file. The model contains unigram counts and bigram probabilities. It populates the `word`, `index`, `unigram_count`, and `bigram_prob` dictionaries based on the model data. If the file cannot be opened, it prints an error message.

3. **`compute_entropy_cumulatively(self, word)`**:
   - Computes the cross-entropy for the current word in the test corpus by calculating the probability using bigram probabilities with linear interpolation. It accumulates the entropy by considering cases where words may be unknown or missing from the model. The cumulative log-probability is updated after each word.

4. **`process_test_file(self, test_filename)`**:
   - Reads and processes the test corpus file word by word. It tokenizes the text, lowercases it, and then calls `compute_entropy_cumulatively` on each token to calculate the cumulative entropy. If the file cannot be opened, it prints an error message.

5. **`main()`**:
   - Parses the command-line arguments, which specify the language model file and the test corpus file. It then creates a `BigramTester` instance, reads the model, and processes the test corpus to calculate the entropy. Finally, it prints the number of words read and the estimated entropy.

### Results 
Testing the model built from a test corpus using a given test corpus.
For instance, to test the model built from small.txt using kafka.txt as a test corpus:

```
$ python BigramTester.py -f small_model.txt -t data/kafka.txt
file small_model.txt has been opened:
Unique word count: 11, total word count: 19
Read 24944 words. Estimated entropy: 13.46
```
comparing this with the provided reference Entropy:

```
$ cat .\entropy_small_kafka.txt
Read 24944 words. Estimated entropy: 13.46
```

Repeating this for remaining combinations:

```
# Model from austen.txt using austen.txt as the austen test corpus
$ python .\BigramTester.py -f .\austen_model.txt -t .\data\austen_test.txt
file .\austen_model.txt has been opened:
Unique word count: 6422, total word count: 130254
Read 10738 words. Estimated entropy: 6.97

$ cat .\entropy_austen_austen.txt
Read 10738 words. Estimated entropy: 6.97
----------------------------------------
# Model from austen.txt using guardian_test.txt as the test corpus
$ python .\BigramTester.py -f .\austen_model.txt -t .\data\guardian_test.txt
file .\austen_model.txt has been opened:
Unique word count: 6422, total word count: 130254
Read 871878 words. Estimated entropy: 9.75

$ cat .\entropy_austen_guardian.txt
Read 871878 words. Estimated entropy: 9.75
----------------------------------------
# Model from austen.txt using kafka.txt as the test corpus
$ python .\BigramTester.py -f .\austen_model.txt -t .\data\kafka.txt
file .\austen_model.txt has been opened:
Unique word count: 6422, total word count: 130254
Read 24944 words. Estimated entropy: 7.36

$ cat .\entropy_austen_kafka.txt
Read 24944 words. Estimated entropy: 7.36
----------------------------------------
# Model from guardian.txt using austen.txt as the test corpus
$ python .\BigramTester.py -f .\guardian_model.txt -t .\data\austen_test.txt
file .\guardian_model.txt has been opened:
Unique word count: 172109, total word count: 8910277
Read 10738 words. Estimated entropy: 6.56

$ cat .\entropy_guardian_austen.txt
Read 10738 words. Estimated entropy: 6.56
----------------------------------------
# Model from guardian.txt using guardian_test.txt as the test corpus
$python .\BigramTester.py -f .\guardian_model.txt -t .\data\guardian_test.txt
file .\guardian_model.txt has been opened:
Unique word count: 172109, total word count: 8910277
Read 871878 words. Estimated entropy: 6.62

$ python BigramTester.py -f guardian_model.txt -t data/guardian_test.txt
Read 871878 words. Estimated entropy: 6.62
----------------------------------------
# Model from kafka.txt using austen_test.txt as the test corpus
python .\BigramTester.py -f .\kafka_model.txt -t .\data\austen_test.txt
file .\kafka_model.txt has been opened:
Unique word count: 2605, total word count: 24944
Read 10738 words. Estimated entropy: 8.73

cat .\entropy_kafka_austen.txt
Read 10738 words. Estimated entropy: 8.73
----------------------------------------
# Model from kafka.txt using small.txt as the test corpus
python .\BigramTester.py -f .\kafka_model.txt -t .\data\small.txt
file .\kafka_model.txt has been opened:
Unique word count: 2605, total word count: 24944
Read 19 words. Estimated entropy: 10.29

cat .\entropy_kafka_small.txt
Read 19 words. Estimated entropy: 10.29
```


### Results Explanation:

The entropy values calculated for different language models and test corpora indicate how well the model fits the test data. Lower entropy suggests that the model is better at predicting the test data (i.e., there is less uncertainty), whereas higher entropy suggests a poor fit.

- **Small Corpus**:
  - Testing the model built from `small.txt` on the `kafka.txt` test corpus yields an entropy of 13.46, indicating a high level of unpredictability (poor model fit for this corpus).

- **Austen Corpus**:
  - Testing the model built from `austen.txt` on the `austen_test.txt` yields an entropy of 6.97, indicating a better fit (model trained on similar text).
  - Testing the same model on `guardian_test.txt` gives a higher entropy (9.75), showing that the model struggles more with this corpus due to differences in writing style and vocabulary.
  - Testing the same model on `kafka.txt` yields an entropy of 7.36, showing that the model is somewhat better at predicting `kafka.txt` compared to the `guardian_test.txt`.

- **Guardian Corpus**:
  - Testing the model built from `guardian.txt` on `austen_test.txt` gives a relatively low entropy of 6.56, indicating a good prediction fit despite differences in corpus content.
  - Testing the same model on `guardian_test.txt` gives a slightly higher entropy of 6.62, indicating that the model is a good fit but still finds some unpredictability.

- **Kafka Corpus**:
  - Testing the model built from `kafka.txt` on `austen_test.txt` yields an entropy of 8.73, indicating a moderate fit but still higher unpredictability due to differences in corpus content.
  - Testing the same model on `small.txt` gives a high entropy of 10.29, suggesting that the model struggles with the limited and different vocabulary in the small test corpus.

### Key Takeaways:
- Models perform best on text similar to their training corpus (e.g., `austen.txt` on `austen_test.txt`).
- Entropy increases when the model is tested on text that diverges from its training data (e.g., `kafka.txt` on `small.txt`).



## Named Entity Recognition

```markdown
### Summary of the Code

The files `BinaryLogisticRegression.py`, and `NER.py` provide the implementation for the Named Entity Recognition (NER) problem using binary logistic regression. The NER model is trained on labeled data to distinguish between "name" entities and "noname" entities using features extracted from tokens. The model can be trained using three types of gradient descent approaches: stochastic, minibatch, and batch gradient descent. After training, the model can be tested on a separate dataset, and the results, including accuracy and confusion matrix, are displayed.

### Summary of the Files

1. **`NER.py`**:
   - This file handles the Named Entity Recognition task. It reads in training and test datasets, extracts features from tokens, and performs training and testing using logistic regression.
   - The model can be trained using different gradient descent techniques: stochastic, minibatch, or batch.

2. **`BinaryLogisticRegression.py`**:
   - This file implements the binary logistic regression model. It includes methods for performing stochastic gradient descent, minibatch gradient descent, and batch gradient descent. The code also includes methods for computing gradients, fitting the model, and classifying test data.

### Function Summaries

#### **`NER.py`**

1. **`NER.__init__(self, training_file, test_file, model_file, stochastic_gradient_descent, minibatch_gradient_descent)`**:
   - Initializes the NER model and sets up feature extraction functions. Depending on the provided arguments, it trains the model using the training data or loads an existing model from a file. It then tests the model on the test dataset.

2. **`NER.capitalized_token(self)`**:
   - Checks if the current token is capitalized.

3. **`NER.first_token_in_sentence(self)`**:
   - Checks if the current token is the first token in a sentence.

4. **`NER.token_contains_digit(self)`**:
   - Checks if the current token contains any digits.

5. **`NER.token_contains_punctuation(self)`**:
   - Checks if the current token contains any punctuation marks.

6. **`NER.read_and_process_data(self, filename)`**:
   - Reads and processes data from the input file. Each line is split into a token and its corresponding label.

7. **`NER.process_data(self, dataset, token, label)`**:
   - Processes a single token-label pair and extracts features for that token. The features and labels are stored in the dataset.

8. **`NER.read_model(self, filename)`**:
   - Reads a pre-trained model from a file.

9. **`NER.main()`**:
   - Main function that parses command-line arguments and starts the NER process.

#### **`BinaryLogisticRegression.py`**

1. **`BinaryLogisticRegression.__init__(self, x=None, y=None, theta=None)`**:
   - Initializes the logistic regression model. It either loads pre-trained weights (`theta`) or initializes weights for training on a new dataset (`x` and `y`).

2. **`BinaryLogisticRegression.sigmoid(self, z)`**:
   - Computes the sigmoid (logistic) function.

3. **`BinaryLogisticRegression.conditional_prob(self, label, datapoint)`**:
   - Computes the conditional probability \( P(\text{label}|\text{datapoint}) \) using the logistic function.

4. **`BinaryLogisticRegression.compute_gradient_for_all(self)`**:
   - Computes the gradient using the entire dataset, used for batch gradient descent.

5. **`BinaryLogisticRegression.compute_gradient_minibatch(self, minibatch)`**:
   - Computes the gradient using a minibatch of data, used for minibatch gradient descent.

6. **`BinaryLogisticRegression.compute_gradient(self, datapoint)`**:
   - Computes the gradient using a single data point, used for stochastic gradient descent.

7. **`BinaryLogisticRegression.stochastic_fit(self)`**:
   - Performs stochastic gradient descent. The model parameters are updated using the gradient computed from a randomly selected data point.

8. **`BinaryLogisticRegression.minibatch_fit(self)`**:
   - Performs minibatch gradient descent. The model parameters are updated using the gradient computed from a minibatch of data points.

9. **`BinaryLogisticRegression.fit(self)`**:
   - Performs batch gradient descent. The model parameters are updated using the gradient computed from the entire dataset.

10. **`BinaryLogisticRegression.classify_datapoints(self, test_data, test_labels)`**:
    - Classifies the test data points and displays the confusion matrix comparing the predicted and actual labels.

11. **`BinaryLogisticRegression.print_result(self)`**:
    - Prints the model parameters and the gradient values.

12. **`BinaryLogisticRegression.update_plot(self, *args)`**:
    - Updates the plot with the current gradient values for visualization.

13. **`BinaryLogisticRegression.init_plot(self, num_axes)`**:
    - Initializes the plot used for visualizing the gradient descent process.

14. **`BinaryLogisticRegression.main()`**:
    - A test function that applies the logistic regression model to a toy dataset.
```

### Results

```
$ python NER.py -d data/ner_training.csv -t data/ner_test.csv -mgd
Results with Minibatch gradient descent

Convergence reached after 56 iterations.
-3.67 5.79 -2.86 2.52 0.25
0.00 -0.00 0.00 0.00 0.00
Model parameters:
0: -3.6739  1: 5.7934  2: -2.8596  3: 2.5157  4: 0.2470
                       Real class
                        0        1
Predicted class:  0 83831.000 3242.000
                  1  879.000 12046.000
				  
=========================================================================
Results with Stochastic Gradient Descent approach:

python NER.py -d data/ner_training.csv -t data/ner_test.csv -s
Working with norm of gradient 0.01835034999040991 at 0 iteration
Working with norm of gradient 0.018737483129725466 at 10 iteration
Working with norm of gradient 0.019141730124714706 at 20 iteration
Working with norm of gradient 0.017271285509742645 at 30 iteration
Working with norm of gradient 1.3167879877564017 at 40 iteration
Working with norm of gradient 0.10642633612864649 at 50 iteration
Working with norm of gradient 0.01866737744107883 at 60 iteration
Working with norm of gradient 0.019099083493097375 at 70 iteration
Working with norm of gradient 0.01714903726381814 at 80 iteration
Working with norm of gradient 0.04906348081923421 at 90 iteration
-4.00 6.19 -3.35 3.83 0.45
0.90 0.90 0.00 0.00 0.00
Model parameters:
0: -3.9990  1: 6.1905  2: -3.3549  3: 3.8257  4: 0.4505
                       Real class
                        0        1
Predicted class:  0 83680.000 3194.000
                  1 1030.000 12094.000
Press Return to finish the program...

				  
==============================================================================
Results with Batch Gradient Descent:

python NER.py -d data/ner_training.csv -t data/ner_test.csv -b
Iter: 1 , Sum of square of Gradient: 0.1314062194642822
Iter: 10 , Sum of square of Gradient: 0.12607190888139588
Iter: 20 , Sum of square of Gradient: 0.12042025810410026
Iter: 30 , Sum of square of Gradient: 0.1150474888496002
Iter: 40 , Sum of square of Gradient: 0.10994218599059213
Iter: 50 , Sum of square of Gradient: 0.10509295741551386
...
...
Iter: 1070 , Sum of square of Gradient: 0.010173228082951959
Iter: 1080 , Sum of square of Gradient: 0.010090122234673345
Iter: 1090 , Sum of square of Gradient: 0.010008725676323792
Iter: 1100 , Sum of square of Gradient: 0.009928975704709004
..
..
Iter: 1260 , Sum of square of Gradient: 0.00883479035045569
..
Iter: 1770 , Sum of square of Gradient: 0.00654534663657462
Iter: 1780 , Sum of square of Gradient: 0.006510127803423647
...
...		  
```

The results presented showcase the performance of a Named Entity Recognition (NER) model using binary logistic regression with three different gradient descent approaches: Mini-batch Gradient Descent, Stochastic Gradient Descent, and Batch Gradient Descent. The evaluation metrics include the number of iterations until convergence, model parameters, and the confusion matrix representing the classification results.

#### **1. Mini-batch Gradient Descent (MBGD)**
- **Convergence**: Convergence was reached after 56 iterations, which indicates that the model learned relatively quickly with mini-batches. This approach balances between the high variance of stochastic gradient descent and the stability of batch gradient descent.
- **Model Parameters**: The model parameters are shown as `[-3.67, 5.79, -2.86, 2.52, 0.25]`. These weights influence how the model classifies entities based on the extracted features.
- **Confusion Matrix**:
  - **Real class 0** (non-name entities): 83,831 instances were correctly classified, while 3,242 were misclassified.
  - **Real class 1** (name entities): 12,046 instances were correctly classified, while 879 were misclassified.
- **Observation**: The Mini-batch Gradient Descent approach yielded a good balance between accuracy and computational efficiency, as seen by the moderate misclassification rates.

#### **2. Stochastic Gradient Descent (SGD)**
- **Convergence**: The training log shows the model working with the gradient norm over multiple iterations. SGD introduces more variability in the learning process due to the randomness of using a single data point at each step, leading to occasional fluctuations in the gradient norm.
- **Model Parameters**: The parameters after training were `[-4.00, 6.19, -3.35, 3.83, 0.45]`. These weights reflect the features learned by the model, which slightly differ from the Mini-batch Gradient Descent parameters.
- **Confusion Matrix**:
  - **Real class 0**: 83,680 correct classifications, 3,194 misclassifications.
  - **Real class 1**: 12,094 correct classifications, 1,030 misclassifications.
- **Observation**: Although the final performance of the model is comparable to Mini-batch Gradient Descent, the variability of the training process in SGD can lead to less stable convergence, requiring more iterations to reach a similar level of performance.

#### **3. Batch Gradient Descent (BGD)**
- **Convergence**: Batch Gradient Descent took much longer to converge, requiring a large number of iterations. The training log shows that even after 1,260 iterations, the sum of the square of the gradient was still decreasing, but very slowly.
- **Observation**: Batch Gradient Descent calculates the gradient using the entire dataset, leading to more stable updates but slower convergence. This approach is less efficient in scenarios with large datasets, as it requires computing the gradient across the entire dataset for each update. It is clear from the logs that the training process is computationally expensive and time-consuming.
- **Performance**: Due to the expected long time of the conclusion of training, the run was aborted after 1.5 hours and 1780 iterations, but it is assumed that the final performance of BGD is comparable to the other methods. However, the time taken to converge is significantly higher, making this approach less practical for large-scale NER tasks.

### **Key Insights:**
- **Efficiency vs. Performance**: Mini-batch Gradient Descent provides a good trade-off between convergence speed and performance, making it an efficient choice for training large models. It combines the benefits of both stochastic and batch approaches.
- **Stochastic Gradient Descent**: SGD can converge faster due to its frequent updates but is more susceptible to fluctuations in the gradient, leading to instability in some cases. Nevertheless, it can still perform well in practice.
- **Batch Gradient Descent**: Although BGD is conceptually straightforward and stable, it is computationally expensive and slower to converge, particularly on large datasets. The results suggest that BGD may not be the most practical approach for training large-scale NER models.

### **Recommendation**:
Given the balance between performance and training efficiency, Mini-batch Gradient Descent is recommended for Named Entity Recognition tasks, especially when dealing with larger datasets. While Stochastic Gradient Descent can be a good alternative for faster convergence, it may require careful tuning and monitoring of the gradient norm to ensure stable training.

