Transformers are the de facto architecture for a wide variety of tasks, including Neural Machine Translation (NMT). They represent a significant improvement over earlier architectures such as Recurrent Neural Networks (RNNs). In this post, we will delve deeper into how RNNs handled NMT and the improvements introduced by Transformers.

# Brief look into RNNs and Transformers

## Recurrent Networks (RNNs)
Before the transformers model, RNN-based architecture, like LSTMs (Long Short-Term Memory) or GRUs (Gated Recurrent Unit) were widely used for machine translation. RNNs process inputs sequentially, one word at a time. 

RNNs have a looping mechanism where information from previous steps is fed back into the network, This allows them to maintain a form or "memory" of past inputs. Specifically:
- RNNs process input sequences one element at a time, in order.
- At each step, the network takes the current input and the previous hidden state as inputs.
- The output at each step depends on both the current input and the accumulated information from all previous inputs.

In translation, an RNN would take one word of a sentence at a time, remember the context of previous words, and try to predict the next word in the translated sentence. 
This sequential processing introduces two challenges:

- **Long term dependency issues**: As the sequences grows longer, it becomes harder for the network to remember the earlier context, leading to a degraded performances. 
- **Lack of parallelism**: Since RNNs process one word at a time, they can't handle the entire sequences in parallel, making training and inferences slow.


## Transformers
The transformer model is a significant improvement over RNNs, especially for translation tasks, for the following reasons:
- **Attention mechanism**: Transformers rely on a mechanism called self-attention. Instead of processing one word at a time like RNNs, the transformer looks at all the words in the input sequence at the same time and weighs the importance of each word relative to others using the attention mechanism. This allows the model to build context for every word in the sentence by considering how it relates to all other words, rather than just processing words sequentially.
- **No recurrence**: Unlike RNNs, the transformer does not rely on recurent structures that process information step by step. Instead it operates on the entire sequence at once, making the model more efficient and capable of handling longer contexts.
- **Parallelization**: Since transformers process all words at once (thanks for self-attention), the model is much more parallelizable. This speeds up training significantly and makes the model scalable to larger datasets and tasks.
- **Handling complex context**: In translation, a word in one language might map to multiple words in another, or its meaning might depend heavily on the other surrounding words (context). The transformer with its self-attention layers, can track these intricate dependencies better than RNNs because it computes relationships between all words in the sequence at once.

## Deeper Look into RNNs
To understand how RNNs handle NMT, we need to explore the architecture, the way it processes sequences over time, and how it overcomes (or fails to overcome) challenges like handling context and dependencies. We'll follw through an example of translating the English Sentence  "I am eating", in French ("Je manage") using the RNN-based architecture.

This explanation will dive into the following topics:

- **Basic RNN architecture** and forward pass.
- **Sequence-to-sequence (Seq2Seq)** model for translation.
- **Challenges** with standard RNNs in translation.
- Improvement made by **LSTM and GRU**.
- The role of **attention** in enhancing RNN-based NMT models.

### 1. Basic RNN Architecture and Forward Pass

An **RNN** is designed to handle sequential data by maintaining a hidden state that evolves as it processes each element in the sequence. The idea is that this hidden state captures the context from previous words to inform the processing of future words. In our example, the **input sequence** is "I am eating" in English.

The RNN processes the input word by word, and at each time step $t$, the hidden state $h_t$ stores a summary of the words seen so far. For our sentence, the RNN processes three words, "I", "am", and "eating".

#### Equations for an RNN
1. **Hidden state updates**:
```math
   h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b_h)
```
   - $h_t$ is the hidden state at time step $t$.
   - $x_t$ is the input at time $t$, typically a word embedding in NMT.
   - $h_{t-1}$ is the hidden state from the previous time step.
   - $W_{hx}$ is the weight matrix connecting the input to the hidden state.
   - $W_{hh}$ is the weight matrix connecting the previous hidden state to the current hidden state.
   - $b_h$ is the bias term, and $tanh$ is the non-linearity applied to the sum.

- In our case:
   - For the word "I", $x_1$ represents the embedding for "I".
   - For the word "am", $x_2$ represents the embedding for "am".
   - For the word "eating", $x_3$ represents the embedding for "eating".
 
   Each word embedding $x_t$ is processed in order to update the hidden state $h_t$, which evolves with each input word.

2. **Output (translation prediction)**:

The output at time $t$, $o_t$ could be generated as:
```math
  o_t = W_{ho}h_t + b_o
```
  where:
  - $W_{ho}$ is the weight matrix that connects the hidden state to the output.
  - $b_o$ is the output bias term.

  The final hiden state $h_3$ at the end of the sentence represents a summary of the entire input sentence "I am eating".
  
  The hidden state evolves like this:
  - $h_1$ stores information after processing "I".
  - $h_2$ stores information after processing "I am".
  - $h_3$ stores information after processing "I am eating".

This process repeats for each word in the sequence, with the hidden state evolving over time to "remember" the context of previous words.

#### Backpropagation Through Time (BPTT)

Training RNNs involves backpropagation through time (BPTT), where the error is propagated backward across all time steps. The issue arises when trying to maintain context across long sequences: 
  - gradients tend to either vanish (become too small), or
  - explode (become too large)
when backpropagating over many time steps.

### 2. Sequence-to-Sequence (Seq2Seq) Model for Translation
In machine translation, we use a special RNN-based architecture called a Sequence-to-Sequence (Seq2Seq) model. This consists of two parts:
   - **Encoder**: Reads the input sequence (source language sentence) and encodes it into a fixed-size context vector.
   - **Decoder**: Uses this context vector to generate the output sequences (translated sentence in the target language).

#### Encoder
The encoder is an RNN (or GRU/LSTM) that processes the input sequence $X = [x_1, x_2, \dots, x_T]$ and produces a sequence of hidden states $[h_1, h_2, \dots, h_T]$. The final hidden state $h_T$ is a summary of the entire sequence and acts as the context vector.

In our case, the encoder reads the source sentence ("I am eating") and encodes it into a fixed-size vector (the final hidden state $h_3$), which acts as the **context vector** $h_T$ summarizing the English sentence.

#### Decoder
The decoder is another RNN (or GRU/LSTM) that takes the context vector $h_T$ ($h_3$ in our example) from the encoder and generates the translated sequence $Y = [y_1, y_2, \dots, y_{T'}]$, one word at a time. At each time step $t$, the decoder uses its own hidden state $s_t$ and the previous word $y_{t-1}$ to generate the next word. Here's how it works:
1. The context vector $h_3$ is passed as the initial hidden state for the decoder.
2. At each time step, the decoder generates a word from the target language vocabulary:
   - at $t = 1$, the decoder predicts "Je" (the first word in French).
   - At $t = 2$, based on the hidden state updated from predicting "Je", it predicts "manage" (the second word in French).

Thus, the decoder outputs "Je manage" as the translation of "I am eating".

Decoder Equation:
```math
s_t = \tanh(W_{sy} y_{t-1} + W_{sh} s_{t-1} + W_{sc} h_T + b_s)

```
Where:
- $s_t$ is the decoder hidden state at time $t$.
- $y_{t-1}$ is the embedding of the previously generated word.
- $W_{sc}$ is a weight matrix connectiving the encoder context vector to the decoder.


The decoder's output is passed through a softmax layer to produce a probability distribution over the target vocabulary:
```math
p(y_t | y_{t-1}, s_t) = \text{softmax}(W_{so} s_t + b_o)
```

### 3. Challenges with Standard RNNs in Translation
While the Seq2Seq architecture is powerful, **RNNs** struggle with some key challenges in machine translation:
- **Long-term dependencies**: As the input sequence grows longer, it becomes harder for RNNs to retain information about words that appeared earlier in the sentence. For example, translating a long sentence requires understanding the subject that might have appeared much earlier.
- **Fixed-size context vector**: The encoder compresses the entire input sequence into a fixed-size vector $h_T$, regardless of the sequence length. For longer sentences, this can be a bottleneck, as important information might get lost in compression.

### 4. Improvements with LSTM and GRU
LSTMs introduce _three gates_: **input gate**, **forget gate**, and **output gate** allowing the network to control the flow of information through its hidden state and memory cell.

- **Input gate**
```math
  i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
```
   Decides which new information to store.
- **Forget gate**
```math
  f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
```
   Decides which information from the previous state should be discarded.
- **Memory update**
```math
  C_t = f_t * C_{t-1} + i_t * \tanh(W_c [h_{t-1}, x_t] + b_c)
```
   Updates the memory cell by adding new information and forgetting the irrelevant parts.
- **Output gate**
```math
  o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
```
   Determines the output based on the updated memory.

GRUs simplify this further by combining the forget and input gates into a single **update gate**.

### 5. Attention Mechanism
One of the biggest issues with RNN-based Seq2Seq model is the **fixed-size context vector**. The **attention mechanism** solves with by allowing the decoder to "attend" to different parts of the input sequence at each decoding step, instead of relying solely on the context vector $h_T$.

At each time step $t$, the decoder computes an attention score for each input hidden state $h_i$:
```math
  a_t^i = \text{softmax}(h_i^\top W_a s_t)
```
These attention scores $a_t^i$ are used to compute a weighted sum of the encoder hidden states, which becomes the **context vector** for that deocding step:
```math
  c_t = \sum_i a_t^i h_i
```

The decoder can then generate the next word based on this dynamic context $c_t$, which allows it to focus on different parts of the input sentence as needed.

### Summary of RNN-Based NMT:

1. **RNN-based Seq2Seq** models translate by encoding the source sentence into a context vector and decoding it step by step.
2. **Challenges** include difficulties in capturing long-range dependencies and handling long sequences.
3. **LSTM and GRU** improve over RNNs by adding gating mechanisms for better memory management.
4. The **attention mechanism** allows the model to dynamically focus on different parts of the input, improving translation quality.

While RNNs laid the foundation for NMT, the introduction of **attention** and **transformers** further revolutionized the field by addressing many of these challenges directly.

## Deeper Look into Transformers

Transformers, first introduced in the paper "Attention is All You Need" (Vasvani et al., 2017), are able to capture long-range dependencies in the data without the need for recurrent processing, making them more efficient and often more effective than RNNs, espcially for longer sequences. 

Key components of a Transformer:

|Component|Purpose|Description|
|---------|-------|-----------|
|Encodere-Decoder Structure|Similar to Seq2Seq models but with cruical differences||
|Self-Attention Mechanism|Enables parallel processing|This mechanism computes relationship scores between all pairs of positions in the input sequence, allowing the model to weigh the importance of different parts of the input when producing each part of the output, regardless of their distance in the sequence|
|Positional Encoding|Helps preserve word order|To maintain information about the order of the sequence (word order), which was implicit in RNNs due to their sequential nature, Transformers use positional encodings added to the input embeddings|
|Feedforward Neural Networks|applied to each position independently||
|Layer Normalization and Residual Connection|For stable training||

### 1. Encoder-Decoder Structure in Transformers
Just like RNN-based Seq2Seq models, the transformers also has an **Encoder-Decoder** structure. However, it processes the entire sequence in parallel rather than step-by-step.

- **Encoder**: The encoder takes the input sequence ("I am eating") and encodes it into a set of hidden representations.
- **Decoder**: The decoder takes these hidden representations and generates the target sequence ("Je manage") one token at a time. However, the decoder does this using self-attention and cross-attention mechanism instead of a recurrent structure.

### 2. Self-Attention Mechanism
The self-attention mechanism is the core innovation in the Transformer. It allows the model to focus on different parts of the input sequence at each layer, depending on which words are more important for understanding the context.

#### Attention Computation
For each word in the input sequence, self-attention computes a weighted sum of all other words in the sentence to capture dependencies between them. For example, when processing "eating" in the input sequence, the attention mechanism helps capture relationships with "am" and "I" in parallel, without having to pass through previous hidden states as in RNNs.

##### Key, Query, and Value
Self-Attention uses three vectors for each word: **Key**, **Query**, and **Value**. These are generated by multiplying the input embeddings with learned weight matrices.

Given an input word $x_t$:
- Query vector $Q$ tells the model what to "focus on".
- Key vector $K$ tells the model what to "match" from other words.
- Value vector $V$ contains the information of the word itself.

For each word $i$ in the input sequence:
```math
\text{Attention}(Q_i, K_j, V_j) = \text{softmax}\left( \frac{Q_i K_j^\top}{\sqrt{d_k}} \right) V_j
```
Where:
- $Q_i$ and $K_j$ are the query and key vectors for words $i$ and $j$, respectively.
- $d_k$ is a scaling factor (dimension of the key vectors).

## 3. Positional Encoding
Unlike RNNs, Transformers have no inherent notion of the order of words in the sequence because they process all words in parallel. To overcome this, position encoding is added to the input embeddings to give the model information about the position of each word in the sequence.

```math
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
```
```math
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
```
Where:
- $pos$ is the position of the word (e.g., 1 for the first word, 2 for the second and so on).
- $i$ is the dimension index of the word embedding.
- $d$ is the total embedding dimension (e.g., 512)
- Choice of sin and cos:
  - Cyclic nature: Both sin and cos are periodic function, which means they naturally retain cyclic patterns. This is useful for encoding positions because it allows the positional encodings to capture repeating structures in language, such as recurring word patterns or sentence structures. The cyclic behavior helps model relative positions between tokens, which is crucial in NLP tasks like translation.
  - Distinctiveness: The use of both sine and cosine functions allows every position in the sequence to have a unique positional encoding across all embedding dimensions. By using sine for even dimensions and cosine for odd dimensions, the positional encoding vectors become distinguishable for each position, while ensuring smooth transitions between neighboring positions.
  - Variation across dimensions:  The combination of sine and cosine with varying frequencies across different dimensions ensures that different dimensions represent positional information at different granularities. For example, lower dimensions may capture long-range dependencies (e.g., entire sentences), while higher dimensions capture short-range dependencies (e.g., nearby words). This is achieved by the scaling factor inside the sine and cosine functions (discussed next).
- 10000 is a constant scaling factor, chosen to control the frequencey of the sinusoids, ensuring that different embedding dimensions capture positional information at different granularities.
  - When $i$ (the dimensional index) is small, $10000^{2i/d}$ is large, and the result of $\frac{\text{pos}}{10000^{2i/d}}$ is small, so the sine/cosine changes more slowly (low frequncy), capturing coarse-grained, long-range dependencies.
  - When $i$ is large, $10000^{2i/d}$ is small, and $\frac{\text{pos}}{10000^{2i/d}}$ becomes large, so the sine/cosine changes more rapidly (high frequncy), capturing fine-grained, short-range dependencies.

This approach ensures that position encodings vary across tokens, allowing the model to infer both short-term and long-term dependencies between tokens in the sequence.

## 3. Feedforward Networks
After applying self-attention, the Transformer passes the output through a feedforward neural network (FNN). The FNN is applied independently to each word position and helps capture more complex relationships in the data.

### FNN Equation
```math
\text{FNN}(x) = \text{ReLU}(W_1x+ b_1)W_2 + b_2
```
For example, after the self-attention step, each word's representation (e.g., the enriched representation of "eating") is passed through a feedforward layer to further transform the information. This happens for each word separately.

## 5. Multi-Head Attention
A key enhancement in Transformers is multi-head attention. Instead of computing a single attention distribution, the model computes multiple attention heads in parallel, allowing it to attend to different parts of the sequence simultaneously.

For example, one attention head might focus on the relationship between "I" and "am", while another focuses on "am" and "eating". This helps the model capture more nuanced relationships in the sentence.

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
```
Where each head is an independent self-attention operation.
In practice:
- One attention head may capture that "eating" relates to "am".
- Another head may capture that "I" is the subject and should align with "Je" in the target language.

## 6. Decoder and Cross-Attention
The decoder in the Transformer works similarly to the encoder but also incorporates cross-attention, which allows it to attend to the encoder's output while generating the target sentence.

For each word it generates (e.g., "Je"), the decoder attends not only to previously generated words in French, but also to the entire input sequence ("I am eating") from the encoder. This helps ensure that the generated translation is accurate and contextually relevant.

At each decoding step:
1. Self-attention looks are previously generated words ("Je" when generating "mange").
2. Cross-attention looks at the encoder's output ("I am eating") to inform the next word generation.

For our example:
- At $t=1$, the decoder generates "Je", focusing on "I" in the input.
- At $t=2$, the decoder generates "mange", focussing on "eating" in the input.

# Transformer vs RNN: Key Improvements
1. **Parallelization**
   - RNNs process words sequentially, meaning they struggle with long sentences and cannot fully leverage modern hardware like GPUs.
   - Transformers process entire sequences in parallel, making them faster to train and more efficient.
2. **Long-Term Dependencies**
   - In RNNs, as sentences get longer, it becomes harder to remember earlier words due to vanishing gradients. This is problematic for translating long sentences.
   - Transformers capture long-range dependencies efficiently using self-attention, which can relate any word to any other word in the sequence.
3. **Context Flexibility**
   - In RNNs, the context is compressed into a single fixed-size vector (the final hidden state). This makes it harder for the decoder to use all the information, especially in long sentences.
   - In Transformers, the self-attention mechanism allows the model to access context from all words at every layer, making it much better at preserving and using global context.
