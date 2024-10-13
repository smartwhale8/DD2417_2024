# Assignment 4

## [Named Entity Recognition with GRUs](NER)
In this part of the assignment, [NER.ipynb](NER/NER.ipynb) implements a Named Entity Recognition (NER) model using Bi-directional Gated Recurrent Units (GRUs). The model processes words and their characters to classify entities as either "name" or "noname". The architecture integrates pre-trained word embeddings (GloVe) along with character-level embeddings to capture both semantic and morphological features of the input words. The final goal of the model is to label each token in the input text as either a named entity or not.

### Key NLP concepts covered:
#### **1. Named Entity Recognition (NER):**
   - NER is the task of identifying entities like people, organizations, locations, dates, etc., in text. In this code, the model identifies whether a word is part of a "name" entity or "noname."

#### **2. Word Embeddings (GloVe):**
   - **Word embeddings** are dense vector representations of words that capture semantic meaning. In this code, pre-trained **GloVe embeddings** are used to represent each word in a sentence, allowing the model to benefit from prior knowledge of word meanings and relationships learned from large corpora.

#### **3. Character-level Embeddings:**
   - Words are also represented using character-level embeddings, which capture **morphological features** like capitalization, prefixes, and suffixes. This is especially useful for distinguishing named entities (e.g., "John" vs "john"). A **bi-directional GRU** processes these character embeddings to create a rich representation of each word.

#### **4. Recurrent Neural Networks (RNNs):**
   - RNNs are designed to process sequences of data. The **Gated Recurrent Unit (GRU)**, used in this model, is a type of RNN that is efficient for capturing sequential information. The **bi-directional GRU** processes text in both forward and backward directions, allowing the model to consider both the preceding and succeeding words when making predictions.

#### **5. Sequence Padding:**
   - Since sentences can vary in length, **padding** is necessary to ensure that all sentences in a batch have the same length. This allows the model to efficiently process the data in batches without losing information.

#### **6. Gradient-based Optimization:**
   - The model uses the **Adam optimizer**, a gradient-based optimization technique, to minimize the classification error (cross-entropy loss). This ensures that the model's weights are updated to improve predictions during training.

#### **7. Confusion Matrix:**
   - A **confusion matrix** is used to evaluate the performance of the model by comparing predicted and actual labels. It provides a breakdown of **true positives**, **false positives**, **true negatives**, and **false negatives**, helping assess the accuracy of the model.


## [Translation using RNNs](translate)
In this part of the assignment, [translate.ipynb](translate/translate.ipynb) implements a **Neural Machine Translation (NMT)** system using an **Encoder-Decoder** architecture. The model is designed to translate sentences from English to Swedish. It uses **Recurrent Neural Networks (RNNs)** and **Gated Recurrent Units (GRUs)** as the core building blocks for the encoder and decoder. Additionally, it supports **attention mechanisms** for improved translation by focusing on relevant parts of the input sentence.

### Key NLP Concepts Covered

#### 1. **Encoder-Decoder Architecture:**
   - The encoder-decoder framework is widely used for sequence-to-sequence tasks like translation. The **encoder** processes the input sequence (English sentence) and compresses it into a hidden representation, while the **decoder** generates the output sequence (Swedish sentence) using this hidden representation.

#### 2. **Recurrent Neural Networks (RNNs) and GRUs:**
   - **RNNs** are designed to handle sequential data, such as sentences. **Gated Recurrent Units (GRUs)** are a variant of RNNs that help mitigate the vanishing gradient problem by using gates to control the flow of information through the network. This makes them well-suited for capturing dependencies in long sequences.

#### 3. **Bidirectional RNNs:**
   - The encoder uses a **bidirectional RNN**, which processes the input sequence in both forward and backward directions. This allows the model to capture context from both past and future words in the sentence, improving translation accuracy.

#### 4. **Attention Mechanism:**
   - The **attention mechanism** enhances the decoder’s ability to focus on relevant parts of the input sentence while generating each word in the translation. It computes attention weights to determine which encoder hidden states should contribute most to the current word prediction.

#### 5. **Pre-trained Word Embeddings (GloVe):**
   - The model uses **pre-trained GloVe embeddings** for the source language (English). These embeddings capture semantic relationships between words and help the model generalize better by leveraging prior knowledge of word meanings.

#### 6. **Teacher Forcing:**
   - During training, the model uses **teacher forcing**, where the true previous word is provided to the decoder instead of its own prediction. This speeds up training but is gradually reduced to encourage the model to rely on its own predictions.

## [Character-level Language Model using Transformers](charlm_bonus)
In this part of the assignment, [charlm.ipynb](charlm_bonus/charlm.ipynb) implements a **Character-level Language Model** using the **Transformer architecture**. The model is designed to predict the next character in a sequence based on the previous 32 characters. The model uses **self-attention** mechanisms and **positional embeddings** to capture long-range dependencies between characters in the input sequence, aiming to produce coherent text generation.

### Key NLP Concepts Covered

#### 1. **Character-level Language Modeling:**
   - The task is to predict the next character in a sequence based on previous characters. This is useful for tasks like **text generation**, **spell correction**, and **autocomplete** systems. The model processes text at the character level rather than word level, making it robust to out-of-vocabulary words.

#### 2. **Transformer Architecture:**
   - The model uses a **Transformer architecture**, which is highly effective for processing sequential data like text. Transformers rely on **self-attention** to model relationships between distant positions in a sequence, making them more efficient than traditional RNNs for capturing long-range dependencies.

#### 3. **Self-Attention Mechanism:**
   - **Self-attention** allows the model to weigh the importance of each character in the sequence when predicting the next character. This helps the model capture context from both nearby and distant characters.

#### 4. **Positional Embeddings:**
   - Since the Transformer does not have a built-in notion of sequence order, **positional embeddings** are added to the input to encode the position of each character in the sequence. This enables the model to differentiate between characters at different positions.

#### 5. **Cross-Entropy Loss:**
   - The model is trained using **cross-entropy loss**, which measures the difference between the predicted probability distribution and the true distribution (the correct next character). This loss function is commonly used for classification tasks.






