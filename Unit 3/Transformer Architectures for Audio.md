# The original transformer

The original transformer model was designed to translate written text from one language into another.

![alt text](<images/Org Transformer.PNG>)

Left: Encoder

Right: Decoder

The encoder receives an input, in this case a sequence of text tokens, and builds a representation of it (its features). This part of the model is trained to acquire understanding from the input

The decoder uses the encoder’s representation (the features) along with other inputs (the previously predicted tokens) to generate a target sequence. This part of the model is trained to generate outputs. In the original design, the output sequence consisted of text tokens

There are also transformer-based models that only use the encoder part (good for tasks that require understanding of the input, such as classification), or only the decoder part (good for tasks such as text generation). 
- An example of an encoder-only model is BERT; an example of a decoder-only model is GPT2.

A key feature of transformer models is that they are built with special layers called ```attention layers```. These layers tell the model to pay specific attention to certain elements in the input sequence and ignore others when computing the feature representations

# Using transformers for audio

![alt text](<images/Transformers for Audio.PNG>)

The audio models still have a standard transformer architecture but has a slight modification the input/output side to allow for audio data instead of text

For Automatic Speech Recognition (ASR): The input is speech and the output is text

There are a few different ways to handle audio so it can be used with a transformer. The main consideration is whether to use the audio in its 
- raw form: as a waveform
- to process it as a spectrogram instead

## Model Inputs

The input to an audio model can be either text or sound. ```The goal is to convert this input into an embedding vector that can be processed by the transformer architecture.```

### Waveform Input for ASR

An Automatic Speech Recognition (ASR) model takes audio as input. 

To be able to use a transformer for ASR, we first need to convert the audio into a sequence of embedding vectors somehow

Models such as ```Wav2Vec2``` and ```HuBERT``` use the audio waveform directly as the input to the model
- Audio Waveform: a 1D sequence of floats where each float represents the sampled amplitude at a given time 
- Raw waveform is normalized to help  standardise audio samples 

After normalising, the sequence of audio samples is ```turned into an embedding``` using a small convolutional neural network known as a ```feature encoder```.

![alt text](<images/CNN Feature Encoder.PNG>)

Each of the convolutional layers in this network processes the input sequence, subsampling the audio to reduce the sequence length, until the final convolutional layer outputs a 512-dimensional vector with the embedding for each 25 ms of audio. Once the input sequence has been transformed into a sequence of such embeddings, the transformer will process the data as usual.

### Spectrogram Input for ASR

Downside of using raw waveform as input: Long Sequence lengths

For example, 30s of audio at a sampling are of 16kHz 
- 30 * 16000 = 480 000

Longer sequence lengths require more computations in the transformer model and thus higher memory usage

Therefore, raw audio waveforms are not usually the most efficient form of representing an audio input. By using a spectrogram, we get the same amount of information but in a more compressed form.

![alt text](<images/Spectrogram Input.PNG>)

Models such as ```Whisper``` first convert the waveform into a log-mel spectrogram. 

```Whisper``` splits the audio into 30s segments. The log-mel spectrogram for each segment has shape ```(80,3000)```.
- 80: Number of mel bins. Mel bins represent the number of requency bands the audio is divided into according to the mel scale. So, 80 distinct bands (specific range of frequencies on the mel scale)
- 3000: Sequence length. This refers to the number of slices/time frames in the spectrogram. The 30s audio is divided into 3000 slices/frames

By converting to a log-mel spectrogram we’ve reduced the amount of input data, but more importantly, this is a much shorter sequence than the raw waveform. 

The log-mel spectrogram is then processed by a small CNN into a sequence of embeddings, which goes into the transformer as usual.

In both cases, there is a CNN network in front of the transformer that converts the inputs into embeddings before the transformer performs its operations

## Model Outputs

The transformer architecture outputs a sequence of hidden-state vectors, also known as the output embeddings. Our goal is to transform these vectors into a text or audio output

### Output for ASR

The goal of an automatic speech recognition model is to predict a sequence of text tokens. This is done by adding a language modeling head — typically a single linear layer — followed by a softmax on top of the transformer’s output. 
- This predicts the probabilities over the text tokens in the vocabulary.

To perform the different audio tasks of ASR, TTS, and so on, we can simply swap out the layers that pre-process the inputs into embeddings, and swap out the layers that post-process the predicted embeddings into outputs, while the transformer backbone stays the same.