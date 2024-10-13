### Summary of the Code

The code implements a **Neural Machine Translation (NMT)** system using an **Encoder-Decoder** architecture. The model is designed to translate sentences from English to Swedish. It uses **Recurrent Neural Networks (RNNs)** and **Gated Recurrent Units (GRUs)** as the core building blocks for the encoder and decoder. Additionally, it supports **attention mechanisms** for improved translation by focusing on relevant parts of the input sentence.

### Function Summaries and Their Roles

#### 1. `EncoderRNN` class:
   - **Task:** Encodes the input sentence (in English) into a hidden representation, which is passed to the decoder to generate the translated sentence (in Swedish).
   - **Key Components:**
     - **Embedding Layer:** Transforms input word indices into dense vectors.
     - **RNN/GRU Layer:** Processes the sequence of embeddings and generates a hidden state for each word.
     - **Bidirectionality:** Allows processing the input sequence in both forward and backward directions, providing richer context.
   - **`forward()` method:** Takes a batch of word indices, generates embeddings, and passes them through the RNN/GRU to produce hidden states and the final hidden representation for decoding.

#### 2. `DecoderRNN` class:
   - **Task:** Decodes the hidden representation from the encoder and generates the translated sentence in Swedish.
   - **Key Components:**
     - **Embedding Layer:** Transforms predicted word indices into embeddings for input to the RNN/GRU.
     - **Attention Mechanism:** Computes attention scores to focus on relevant encoder outputs during decoding.
     - **RNN/GRU Layer:** Processes the embeddings and attention context to predict the next word in the target sequence.
   - **`forward()` method:** Takes the previous word and the encoder’s hidden states as input, computes the attention context, and predicts the next word.

#### 3. `TranslationDataset` class:
   - **Task:** Prepares the translation dataset by converting sentences into sequences of word indices.
   - **`__init__()` method:** Tokenizes and converts both source (English) and target (Swedish) sentences into word indices.
   - **`__getitem__()` method:** Retrieves a source-target sentence pair by index, returning both the input and target sequences for training.

#### 4. `pad_sequence()` function:
   - **Task:** Pads all sequences in a batch to the same length for efficient processing in batches.
   - **Process:** Takes a batch of source and target sequences and pads each sentence to the length of the longest sentence in the batch.

#### 5. `evaluate()` function:
   - **Task:** Evaluates the trained model by translating sentences and comparing them with the correct target translations.
   - **Process:** Generates translations for each sentence in the evaluation dataset, computes a confusion matrix, and displays statistics on the number of correctly and incorrectly predicted words and sentences.

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

### High-Level Goal

The overall goal of the code is to build a **sequence-to-sequence translation model** that learns to translate English sentences into Swedish. By combining **RNN/GRU-based encoders and decoders**, **attention mechanisms**, and **pre-trained word embeddings**, the model aims to generate accurate translations while capturing the syntactic and semantic structure of both languages.
