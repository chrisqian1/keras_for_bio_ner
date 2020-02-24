# keras_for_bio_ner
Biomedical named entity recognition using keras
### Introduction
This repository is for biomedical named entity recognition task based on keras. It will read text and entity in the docset and arrange it with aspect of doc, sentence and token level. Also, it would prepare the input, train and test on data using various models in keras. Models avaiable now is Lstm, Lstm_crf and Bert. Other models would be added in the future.
### Model setting
For Lstm and Lstm_Crf model, tokenizers in nltk are used to split sentence and tokenize. The main feature are word vectors (using Glove embeddings). In addition, there's one option of POS feature. (using random embedding). Only one BiLstm layer and one dropout layer are used.
For Bert, nltk sent tokenizer is used to split sentence and bert tokenized is used to tokenize word piece. Word piece is the only features used for training. 
When training, the best model will be saved as the final model. Category accuracy or crf viterbi accuracy will be used to choose the best model by evaluating on validation set.
### usage and Example
Task are trained and tested on NBCI disease corpus. 
To train and test, using command line:
```shell
python nerdocset_ncbi.py --option tv --dataset data --modelname Lstm --batchsize 32 --epoch 20 --embedding Glove --times 5 --experiment 0
```
option: t for train and v for validate(test), required
dataset: folder of the dataset, required
modelname: Lstm, Lstm_Crf or Bert, required
batchsize: batch size used when training, default 32. It should be 8 or less in Bert considering the performance of GPU
epoch: epoches used when training, default 20
embedding: word embedding, can be Glove or Word2Vec
times: total times of training, default 5
experiment: No. of experiment
Reference
