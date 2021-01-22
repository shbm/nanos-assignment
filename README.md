# NANOS ASSIGNMENT

## Prerequisites
* Python 3
* Chromium Browser (Google Chrome) to run selenium for reading webpage content

## Usage

```
$ pip install -r requirements.txt

$ python3 main.py -u https://nanos.ai/ -w digital -w marketing -similar 5
```

## Example

```
$ python3 main.py -u https://nanos.ai/ -w digital -w marketing --similar 5

Downloading webpage: https://nanos.ai/

digital : marketing events latest navigate record
marketing : necessary website online businesses clients
```

## Approach

* It downloads the webpage using selenium with Chromium worker. I tried using packages like request or urllib3. They failed for SPAs due to the restrictions enforced by Angualr, React, Vue.JS etc.
* After downlading the webpage, it performs preprocessing and computes word embeddings. For preprocessing we use html tags removal using regex, words decontraction and stop words removal from the nltk library which supports multiple languages.
* Once it gets proprocessed data a Word2Vec baseed model is trained to compute the word embeddings for all the sentences and words in the webapge. The final step is to use those embeddings to compute cosine distances from all the words in the vocabalary. It sorts the words based on embeddings and returs the most similar words.

## Other Approaches Tried

* I tried using BERT based approach but could not contiue because of the large model size and GPU limitation.
* TF-IDF+SVD based vectorization. I tried it first as it is a computationaly inexpensive method but it failed to produce meaningful results.

## Limitations
* The current model depends on the size of the webpage which will mostly be small. No pretrained model is used because of the time limitaion. However, it can scaled by using a larger pretrained model. It might not give meaningful results on sparse webapges with very little content.

* The results vary slightly with each run since it's being trained from scratch for every run. However it can be fixed by setting a seed value.

### THANK YOU!

I got to learn about a new methods while working on this task. I learned about the limitations of libraries like requests and urllib for web scraping and selenium's power. I also liked writing easily testable and modularized code.