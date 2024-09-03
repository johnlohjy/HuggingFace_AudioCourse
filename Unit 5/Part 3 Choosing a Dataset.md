## Features of speech datasets

1. Number of hours

Number of training hours = How large the dataset is

Bigger datasets aren’t necessarily better. If we want a model that generalises well, we want a diverse dataset with lots of different speakers, domains and speaking styles

2. Domain

The domain entails where the data was sourced from, whether it be audiobooks, podcasts, YouTube or financial meetings. 

Each domain has a different distribution of data. For example, audiobooks are recorded in high-quality studio conditions (with no background noise) and text that is taken from written literature. Whereas for YouTube, the audio likely contains more background noise and a more informal style of speech.

We need to match our domain to the conditions we anticipate at inference time. For instance, if we train our model on audiobooks, we can’t expect it to perform well in noisy environments

3. Speaking style

The speaking style falls into one of two categories:
- Narrated: read from a script
- Spontaneous: un-scripted, conversational speech

The audio and text data reflect the style of speaking. Since narrated text is scripted, it tends to be spoken articulately and without any errors. Whereas for spontaneous speech, we can expect a more colloquial style of speech, with the inclusion of repetitions, hesitations and false-starts

4. Transcription Style

Transcription style refers to whether the target text has punctuation, casing or both. 

If we want a system to generate fully formatted text that could be used for a publication or meeting transcription, we require training data with punctuation and casing. 

If we just require the spoken words in an un-formatted structure, neither punctuation nor casing are necessary. 

In this case, we can either pick a dataset without punctuation or casing, or pick one that has punctuation and casing and then subsequently remove them from the target text through pre-processing

More detailed breakdown of audio datasets: https://huggingface.co/blog/audio-datasets#a-tour-of-audio-datasets-on-the-hub

Using own audio data with HuggingFace Datasets/Creating custom audio dataset: https://huggingface.co/docs/datasets/audio_dataset

