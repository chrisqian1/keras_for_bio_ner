# keras_for_bio_ner
Biomedical named entity recognition using keras
### Introduction
This repository is for biomedical named entity recognition task based on keras. It will read text and entity in the docset and arrange it with aspect of doc, sentence and token level. Also, it would prepare the input, train and test on data using various models in keras. Models avaiable now is Lstm, Lstm_crf and Bert. Other models would be added in the future.
### Model setting
For Lstm and Lstm_Crf model, tokenizer in nltk are used to split sentence and tokenize. The main feature are word vectors (using Glove embeddings). In addition, there's one option of POS feature. (using random embedding). Only one BiLstm layer and one dropout layer are used.
### usage and Example
Task are trained and tested on NBCI disease corpus. 
To train and test, using command line:
```shell
python nerdocset_ncbi.py --option tv --dataset data --modelname Lstm --batchsize 32 --epoch 20 --embedding Glove --times 5 --experiment 0
```
