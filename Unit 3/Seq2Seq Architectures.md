# Seq2Seq Architectures

Encoder-decoder models

This model maps a sequence of one kind of data to a sequence of another kind of data
- the input and output sequences can have different lengths
- makes seq2seq models suitable for NLP tasks such as text summarisation, translation between different languages, audio tasks such as speech recognition

## Automatic Speech Recognition Use Case

![alt text](<images/Whisper AI.PNG>)

```Left side: Transformer Encoder```
- Takes as input a log-mel spectrogram
- Encodes the log-mel spectrogram to form a sequence of encoder hidden states that extract important features from the spoken speech
- Hidden-states tensor represents the input sequence as a whole and effectively encodes the “meaning” of the input speech

```Right side: Transformer Decoder```

The output of the encoder is passed into the transformer decoder using a mechanism called cross-attention. This is like self-attention but attends over the encoder output. 

From this point on, the encoder is no longer needed.

The decoder predicts a sequence of text tokens in an autoregressive manner, a single token at a time, starting from an initial sequence that just has a “start” token in it (SOT in the case of Whisper). 

At each following timestep, the previous output sequence is fed back into the decoder as the new input sequence. 

In this manner, the decoder emits one new token at a time, steadily growing the output sequence, until it predicts an “end” token or a maximum number of timesteps is reached.

2 differences from the architecture of an encoder
- Decoder has a cross-attention mechanism that allows it to look at the encoder's representation of the input sequence
- Decoder's attention cannot look into the future

In this design, the decoder plays the role of a language model, processing the hidden-state representations from the encoder and generating the corresponding text transcriptions

A typical loss function for a seq2seq ASR model is the cross-entropy loss, as the final layer of the model predicts a probability distribution over the possible tokens. 

This is usually combined with techniques such as beam search to generate the final sequence (improve the quality of predictions). 

The metric for speech recognition is WER or word error rate, which measures how many substitutions, insertions, and deletions are necessary to turn the predicted text into the target text — the fewer, the better the score.