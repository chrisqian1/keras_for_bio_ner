from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.initializers import Constant
from keras.optimizers import Adam
from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from gensim.models.keyedvectors import KeyedVectors
import os


def file_line2list(sfilename, fcoding='utf8', lowerID=False, stripID=True, verbose=0):
    finput = open(sfilename, 'r', encoding=fcoding)
    lines = finput.readlines()
    for i in range(len(lines)):
        if stripID: lines[i] = lines[i].strip()
        else: lines[i] = lines[i].strip('\r\n')
        if lowerID: lines[i] = lines[i].lower()
    finput.close()
    if verbose:  print('\nLoading {:,} lines from {} ...'.format(len(lines), sfilename))
    return lines


def load_word_voc_file(filename=None, verbose=0):
    if not os.path.exists(filename): return None
    words = file_line2list(filename, verbose=verbose)
    word_dict = {word:i for i, word in enumerate(words)}
    return word_dict


def load_bert_tokenizer(bert_model_path):
    # load tokenizer and token dict from bert path
    dict_path = os.path.join(bert_model_path, 'vocab.txt')
    token_dict = load_word_voc_file(dict_path, verbose=True)
    tokenizer = Tokenizer(token_dict)
    return token_dict, tokenizer


def load_bert_model(bert_path, verbose=0):
    # load bert model from bert path
    if verbose:
        print('\nLoading the BERT model from {} ...'.format(bert_path))
    config_path = os.path.join(bert_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    return bert_model


def load_pretrained_embedding_from_file(embed_file, word_dict, embedding, EMBED_DIM=100):
    '''
    load pretained embbeding matrix
    '''

    np.random.seed(1234)
    # load glove embedding
    if embedding == 'Glove':
        lines = file_line2array(embed_file, sepch=' ', verbose=True)
        # filter the lines containing words in word_dict
        wlines = [values for values in lines if values[0] in word_dict]
        # initialze embedding matrix
        embed_matrix = np.random.normal(0, 1, (len(word_dict), EMBED_DIM))

    # load word2vec embedding
    else:
        print('start loading embedding')
        # load embedding file
        if os.path.exists('PubMed-and-PMC-w2v.txt'):
            wlines = open('PubMed-and-PMC-w2v.txt', encoding = 'utf-8').readlines()
            wlines = [line.strip('\n').split(' ') for line in wlines]
            EMBED_DIM = len(wlines[0])-1
        else:
            f = open('PubMed-and-PMC-w2v.txt', 'w')
            word_vector = KeyedVectors.load_word2vec_format(embed_file, binary=True)
            print(len(word_vector.vocab))
            wlines = []
            # load word vectors
            for word in word_vector.vocab:
                if word in word_dict:
                    line = [word] + [str(vector) for vector in word_vector[word].tolist()]
                    wlines.append(line)
                    f.write(' '.join(line) + '\n')
            EMBED_DIM = word_vector.vector_size

        print('end loading embedding')
        scope = np.sqrt(3. / EMBED_DIM)
        # initialze embedding matrix
        embed_matrix = np.random.uniform(-scope, scope, size=(len(word_dict), EMBED_DIM)).astype('float32')

    # add embedding weights to matrix
    for values in wlines:
        idx = word_dict[values[0]]
        embed_matrix[idx] = np.asarray(values[1:], dtype='float32')
    return embed_matrix, EMBED_DIM


def create_bert_classification_model(bert_path, level, modelname, num_classes):
    '''
    create bert model
    level: sent, token
    '''

    # model input
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    # bert model layer
    bert_model = load_bert_model(bert_path)
    for l in bert_model.layers: l.trainable = True
    x = bert_model([x1_in, x2_in])

    # crf and dense layer accroding to crf and level
    if level == 'token':
        if 'Crf' in modelname:
            x = TimeDistributed(Dense(50, activation="relu"))(x)
            output = CRF(num_classes)(x)
        else:
            output = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    else:
        x = Lambda(lambda x: x[:, 0])(x)
        if 'Crf' in modelname:
            x = Dense(50, activation='relu')(x)
            output = CRF(num_classes)(x)
        else:
            output = Dense(num_classes, activation='softmax')(x)

    # build and compile model
    model = Model([x1_in, x2_in], output)
    if 'Crf' in modelname:
        model.compile(loss=crf_loss, optimizer=Adam(1e-5), metrics=[crf_viterbi_accuracy])
    else:  
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['categorical_accuracy'])
    model.summary()
    return model


def create_lstm_classification_model(level, modelname, vocab_size, POS_size, num_classes, EMBED_MATRIX, embed_dim):
    '''
    cread bilstm model
    level: sent, token
    '''

    # model input
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    # embedding layer
    word_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=Constant(EMBED_MATRIX),
                           input_length=None)(x1_in)
    if 'POS' in modelname:
        POS_embed = Embedding(input_dim=POS_size, output_dim=20, input_length=None)(x2_in)
        word_embed = Concatenate()([word_embed, POS_embed])
    # dropout layer
    word_embed = Dropout(0.2)(word_embed)
    # bilstm layer
    x = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(word_embed)

    # crf and dense layer accroding to crf and level
    if level == 'token':
        if 'Crf' in modelname:
            x = TimeDistributed(Dense(50, activation="relu"))(x)
            output = CRF(num_classes)(x)
        else:
            output = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    else:
        x = Lambda(lambda x: x[:, 0])(x)
        if 'Crf' in modelname:
            x = Dense(50, activation='relu')(x)
            output = CRF(num_classes)(x)
        else:
            output = Dense(num_classes, activation='softmax')(x)

    # build and compile model
    model = Model([x1_in, x2_in], output)
    if 'Crf' in modelname:
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.summary()
    return model