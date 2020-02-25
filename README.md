# keras_for_bio_ner:Biomedical named entity recognition using keras
### Introduction
This repository is for biomedical named entity recognition task using various models based on keras. It reads text and entity information in the biomedical literature with abstract-level annotations, like NCBI disease corpus. The project prepares the input for various models, train and test on data using these models supported in keras. Models available now are Lstm, Lstm_Crf, Bert and Bert_Crf. Other models will be added in the future.
### Model setting
For Lstm and Lstm_Crf model, the nltk toolkit is used to split abstracts into sentences and tokenize these sentences into tokens. The main feature is pre-trained word vectors (Glove or Word2Vec). Additional features include POS etc.

For Bert and Bert_Crf, the nltk toolkit is used to only split sentences and BERT tokenizer is used to tokenize a sentence into word pieces. BioBert pre-trained model based on PubMed is loaded as initial model.

When training, the best model on the validation set will be saved as the final model. Category accuracy or crf-viterbi-accuracy when CRF is involved will be used to choose the best model by evaluating on the validation set.
### Usage and Example
Training and testing are performed on the NBCI disease corpus [1]. To train and test, using the following command line:
```shell
python bio_ner.py --operation tv --dataset data --modelname Lstm --batchsize 32 --epochs 20 --pre-trained Glove --times 5 --experiment 0
```
option: t for train and v for validate(test), required

dataset: folder of the dataset, required

modelname: Lstm, Lstm_Crf, Bert and Bert_Crf, required

batchsize: batch size used when training, default 32. It should be 8 or less in Bert when considering the capacity of GPU.

epochs: epochs used when training, default 20.

embedding: pre-trained word embedding, can be Glove or Word2Vec.

times: total times of training, default 5

experiment: No. of experiment

### Experimental results on the NCBI disease corpus
| Model    | Precision | Recall | F1    |
| -------- | --------- | ------ | ----- |
| Lstm     | 81.99     | 78.31  | 80.09 |
| Lstm_Crf | 80.99     | 81.45  | 81.22 |
| Bert     | 80.51     | 85.54  | 82.95 |
| Bert_Crf | 83.08     | 85.62  | 84.31 |

### Reference
[1] Habibi,M. et al. (2017) Deep learning with word embeddings improves biomedical named entity recognition. Bioinformatics, 33, i37–i48.

[2] Thanh Hai Dang, Hoang-Quynh Le, Trang M Nguyen, and Sinh T Vu. 2018. D3ner: Biomedical named entity recognition using crf-bilstm improved with finetuned embeddings of various linguistic information. Bioinformatics.

[3] Wang, X., Zhang, Y., Ren, X., Zhang, Y., Zitnik, M., Shang, J., Langlotz, C., and Han, J. (2018). Cross-type biomedical named entityrecognition with deep multi-task learning. CoRR, abs/1801.09851.

[4] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2019. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. arXiv:1901.08746 [cs]. ArXiv: 1901.08746.

[5] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for
word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532–
1543.

[6] Noah A Smith. Contextual word representations: A contextual introduction. arXiv
preprint arXiv:1902.06006, 2019.

[7] Python packages used: keras: https://github.com/keras-team/keras; keras-bert: https://github.com/CyberZHG/keras-bert; keras-xlnet: https://github.com/CyberZHG/keras-xlnet; keras-contrib: https://github.com/keras-team/keras-contrib
