from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from ie_model import *
from nerdoc_ncbi import *
import numpy as np
import random
import os
from argparse import ArgumentParser

random.seed(1234)
max_seq_len = 128
max_word_len = 30


class nerDocSet():
    def __init__(self, id = 'train'):
        self.id = id        
        self.doc_dict = {}
        self.X = None
        self.y = None
        self.X_v = None
        self.y_v = None
        self.match_dict = {}    
        self.type_list = []
        
    def load_docset(self):
        '''
        load text and entity file.
        '''

        # read from file
        text_file = self.id + '.txt'
        entity_file = self.id + '.ent'
        text_input = open(text_file, encoding='utf-8')
        text_list = text_input.readlines()
        entity_input = open(entity_file, encoding='utf-8')
        entity_list = entity_input.readlines()

        # generate text input
        for text in text_list:
            docid = text.split('\t')[0]
            doc_content = ' '.join(text.split('\t')[1:])
            doc = nerDoc(doc_content)
            doc.docid = docid
            self.doc_dict[docid] = doc

        # generate entity input
        for entity in entity_list:
            em = EntityMention(entity.split('\t'))
            self.doc_dict[em.docid].emlist.append(em)
        return

    def preprocess(self, tokenizer, label_schema):
        '''
        preprocess docset
        '''

        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            # tokenize
            doc.split_sentence(tokenizer)
            # transfer entity position from doc level to sent level
            doc.align_doc_to_snt()
            doc.transfer_entity_from_doc_to_snt()
            # assign features and label
            for sent in doc.sntlist:
                sent.assign_labels(label_schema)
                if tokenizer == None:
                    sent.add_POS()
        return

    def update_label_dict(self, dict_list):
        '''
        update index dicts
        '''

        word_dict, POS_dict, label_dict = dict_list

        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            for sent in doc.sntlist:
                for token in sent.tokenlist:
                    # update word dict
                    if len(word_dict) != 0:
                        if token.word not in word_dict:
                            word_dict[token.word] = len(word_dict)
                    # update POS dict
                    if len(POS_dict) != 0:
                        if token.POS not in POS_dict:
                            POS_dict[token.POS] = len(POS_dict)
                    # update label dict
                    if token.label not in label_dict:
                        label_dict[token.label] = len(label_dict)
        return

    def prepare_model_input(self, modelname, tokenizer, dict_list, validate):
        '''
        prepare the input of the model
        '''

        word_dict, POS_dict, label_dict = dict_list

        index_array, segment_array, self.y = [], [], []
        # randomly select 10% of data as validation set
        if validate:
            index_array_v, segment_array_v, self.y_v = [], [], []
            sent_count = 0
            for docid in self.doc_dict:
                doc = self.doc_dict[docid]
                sent_count += len(doc.sntlist)
            random_list=random.sample(range(sent_count),sent_count//10)
            print(sent_count)

        # get features and label from each sentence
        # use sent index to select validation data
        sent_i = 0
        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            for sent in doc.sntlist:
                # get feature index from sent
                if 'Bert' in modelname:
                    index, segment = tokenizer.encode(sent.text, max_len = max_seq_len)
                    index = [item for item in index]
                    segment = [item for item in segment]
                else:
                    word_ix_seq = [word_dict[token.word] for token in sent.tokenlist][:max_seq_len]
                    index = word_ix_seq + [0] * (max_seq_len - len(word_ix_seq))
                    POS_ix_seq = [POS_dict[token.POS] for token in sent.tokenlist][:max_seq_len]
                    segment = POS_ix_seq + [0] * (max_seq_len - len(POS_ix_seq))

                # get label index from sent
                label_ix_seq = [label_dict[token.label] for token in sent.tokenlist][:max_seq_len]
                label_ix_seq = label_ix_seq + [0] * (max_seq_len - len(label_ix_seq))
                # transfer label index to one hot encoding
                label_onehot = to_categorical(label_ix_seq, len(label_dict))

                # add features and label to array
                if validate and sent_i in random_list:
                    index_array_v.append(index)
                    segment_array_v.append(segment)
                    self.y_v.append(label_onehot)
                else:
                    index_array.append(index)
                    segment_array.append(segment)
                    self.y.append(label_onehot)

                sent_i+=1

        # transfer data to numpy arrays
        if validate:
            index_array_v, segment_array_v, self.y_v = \
                np.array(index_array_v), np.array(segment_array_v), np.array(self.y_v)
            self.X_v = [index_array_v, segment_array_v]
        index_array, segment_array, self.y = \
            np.array(index_array), np.array(segment_array),  np.array(self.y)
        self.X = [index_array, segment_array]
        return

    def train(self, modelname, dict_list, bert_path, model_path, validate, batch_size, epoch, embedding='Glove'):
        '''
        train model with data
        '''

        word_dict, POS_dict, label_dict = dict_list

        # create initial model according to modelname
        if 'Bert' in modelname:
            model = create_bert_classification_model(bert_path, 'token', modelname, len(label_dict))
        else:
            # get word embedding path
            if embedding == 'Glove':
                embed_path = '../../AnaPython/glove/glove.6B.100d.txt'
            else:
                embed_path = '../../AnaPython/glove/PubMed-and-PMC-w2v.bin'
            # get word embedding matrix and dimension
            if 'Lstm' in modelname:
                embed_matrix, embed_dim = load_pretrained_embedding_from_file(embed_path, word_dict, embedding)
            else:
                embed_matrix, embed_dim = None, 0
            model = create_lstm_classification_model( 'token', modelname, len(word_dict), len(POS_dict),
                                                     len(label_dict), embed_matrix, embed_dim)

        # use checkpoint to select best model during training
        monitor = 'val_crf_viterbi_accuracy' if 'Crf' in modelname else 'val_categorical_accuracy'
        checkpointer = ModelCheckpoint(filepath=model_path, monitor = monitor, verbose=1, save_best_only=True)
        # use validation set to evaluate checkpoint model
        validation = (self.X_v, self.y_v) if validate else None
        # fit the model
        model.fit(x = self.X, y = self.y, batch_size=batch_size, epochs=epoch, validation_data=validation, callbacks=[checkpointer])
        print('model_saved')
        return

    def validate(self, modelname, dict_list, bert_path, batch_size, model_path, output_path, embedding='Glove'):
        '''
        test/validate with saved model and evaluate the result
        '''

        word_dict, POS_dict, label_dict = dict_list

        # create initial model according to modelname
        if 'Bert' in modelname:
            model = create_bert_classification_model(bert_path, 'token', modelname, len(label_dict))
        else:
            # get word embedding path
            if embedding == 'Glove':
                embed_path = '../../AnaPython/glove/glove.6B.100d.txt'
            else:
                embed_path = '../../AnaPython/glove/PubMed-and-PMC-w2v.bin'
            # get word embedding matrix and dimension
            if 'Lstm' in modelname:
                embed_matrix, embed_dim = load_pretrained_embedding_from_file(embed_path, word_dict, embedding)
            else:
                embed_matrix, embed_dim = None, 0
            model = create_lstm_classification_model('token', modelname, len(word_dict), len(POS_dict),
                                                     len(label_dict), embed_matrix, embed_dim)
        print('create_test_model')

        # load model weights from saved model
        model.load_weights(model_path)
        print('model_load')
        # predict and evaluate
        pre = model.predict(self.X, batch_size=batch_size)
        output = open(output_path, 'w')
        self.evaluate(pre, label_dict, output)
        return

    def evaluate(self, predict_value, label_dict, output):
        '''
        evaluate the result
        '''

        self.match_dict = {}
        out = open('tmp1_1.txt','w')

        # use sent index to assign predicted value to each sentence
        sent_i = 0
        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            for sent in doc.sntlist:
                # assign recognized tags to tokens
                rlabels = predict_value[sent_i]
                sent.assign_rlabels(rlabels, label_dict)
                sent_i += 1

            # extract recognized entity in each sentence
            doc.extract_entity_mention()
            for sent in doc.sntlist:
                for token in sent.tokenlist:
                    out.write('{}\t{}\t{}\n'.format(token.word, token.label, token.rlabel))
                out.write('\n')

            for sent in doc.sntlist:
                # extract gold and recognized entity position in the sent
                gold_dict, pred_dict = sent.get_entity_position_dict()
                # match and count the type of the gold and recognized entity
                self.match_gold_test_entity(gold_dict, pred_dict)

        # generate type list
        self.generate_type_list(label_dict)
        # generate and print confusion matrix
        self.generate_confusion_matrix(output)

        # remove O if it's in the type list
        if 'O' in self.type_list:
            self.type_list.remove('O')
        # print title
        title = 'Type\tGold\tTP+FN\tTP\tP\tR\tF1'
        print(title)
        output.write(title+'\n')

        # for each type, output precision, recall and f1
        for type in self.type_list:
            self.output_prf([type], type, output)
        # output the total precision, recall and f1
        self.output_prf(self.type_list, 'Total', output)
        return

    def match_gold_test_entity(self, gold_dict, rem_dict):
        '''
        match gold entity with recognized entity
        '''

        for sepos in gold_dict:
            if sepos in rem_dict:
                # if two entity pos matches, match their type
                match_key = '{}-{}'.format(gold_dict[sepos], rem_dict[sepos])
            else:
                # if only gold entity pos exists, match type with O
                match_key = '{}-O'.format(gold_dict[sepos])
            update_dict_count(self.match_dict, match_key)

        for sepos in rem_dict:
            if sepos not in gold_dict:
                # if only predicted entity pos exists, match type with O
                match_key = 'O-{}'.format(rem_dict[sepos])
                update_dict_count(self.match_dict, match_key)
        return

    def generate_type_list(self, label_dict):
        '''
        generate type list from label dict
        '''

        self.type_list = []
        for key in label_dict:
            # generate and append type
            type = key.split('-')[-1]
            if type != 'O' and type not in self.type_list:
                self.type_list.append(type)
        # append type O
        self.type_list.append('O')
        return

    def generate_confusion_matrix(self, output):
        '''
        create and print confusion matrix
        '''

        # create initial matrix
        l = len(self.type_list)
        conf_matrix = np.zeros([l, l])
        # print matrix title
        matrix_title = '\t'+'\t'.join(self.type_list)
        print(matrix_title)
        output.write(matrix_title+'\n')

        for i,type1 in enumerate(self.type_list):
            for j,type2 in enumerate(self.type_list):
                key = '{}-{}'.format(type1,type2)
                # find count value for each match in match dict
                if key in self.match_dict:
                    value = self.match_dict[key]
                else:
                    value = 0
                conf_matrix[i][j] = value

            # generate and print one row of the matrix
            row = [type1]+[str(item) for item in conf_matrix[i].tolist()]
            print('\t'.join(row))
            output.write('\t'.join(row)+'\n')
        return

    def output_prf(self, type_list, type_name, output):
        '''
        calculate and print prf
        '''

        tp = tp_fp = gold = 0

        # calculate tp, tp+fp and gold from match_dict
        for key in self.match_dict:
            # key1:gold_type key2:pred_type
            key1, key2 = key.split('-')
            # count according to key1 and key2
            if key1 == key2 and key1 in type_list:
                tp += self.match_dict[key]
            if key1 in type_list:
                gold += self.match_dict[key]
            if key2 in type_list:
                tp_fp += self.match_dict[key]

        # calculate p, r, f
        p = round(tp/tp_fp*100, 2) if tp_fp != 0 else 0
        r = round(tp/gold*100, 2) if gold != 0 else 0
        f = round(2*p*r/(p+r), 2) if p != 0 and r != 0 else 0

        # print p, r, f
        result = '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(type_name, gold, tp_fp, tp, p, r, f)
        print(result)
        output.write(result+'\n')
        return

    def test_output(self, testout_path):
        '''
        output sentence with FP or FN error
        '''

        # add entity tags to the sentence
        def write_to_output(emlist, tokenlist):
            sent_text = []
            last_pos = 0
            for em in emlist:
                # tokens before entity
                for i in range(last_pos, em.spos):
                    sent_text.append(tokenlist[i].word)
                # entity start tag
                sent_text.append('||{')
                # tokens in entity
                for i in range(em.spos, em.epos+1):
                    sent_text.append(tokenlist[i].word)
                # entity end tag
                sent_text.append('}||')
                last_pos = em.epos + 1
            # tokens after last entity
            for i in range(last_pos, len(tokenlist)):
                sent_text.append(tokenlist[i].word)
            # return the whole sentence
            return ' '.join(sent_text)+'\n\n'

        output = open(testout_path, 'w')
        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            output.write('{}\n'.format(docid))
            for sent in doc.sntlist:
                # add gold entity tag and predicted entity tag to sentence
                gold_sent = write_to_output(sent.emlist, sent.tokenlist)
                pred_sent = write_to_output(sent.remlist, sent.tokenlist)
                # output if two sentence do not match
                if gold_sent != pred_sent:
                    output.write(gold_sent)
                    output.write(pred_sent)
                    output.write('\n')
        return

def update_dict_count(dic, key):
    # add one to the value if key exist
    if key in dic:
        dic[key] += 1
    # initial key if it doesn't exist
    else:
        dic[key] = 1

def generate_dict_list(modelname):
    if 'Bert' in modelname:
        return {}, {}, {'O': 0}
    else:
        return {'[pad]': 0}, {'[pad]': 0}, {'O': 0}


if __name__=="__main__":
    # use argument parser to receive command options
    # python nerdocset_ncbi.py --option tv --datapath data --modelname Lstm_Crf --batchsize 32 --epoch 5 --stimes 1 --times 2 --experiment 0
    parser = ArgumentParser()
    parser.add_argument('--option', required=True, help='t | v | tv', default='tv')
    parser.add_argument('--datapath', required=True, help='data', default='data')
    parser.add_argument('--modelname', required=True, help='Bert | Lstm | Crf', default='Lstm')
    parser.add_argument('--batchsize', type=int, help='number of batchsize', default=32)
    parser.add_argument('--epoch', type=int, help='number of total epoches', default=20)
    parser.add_argument('--embedding', type=str, help='word embedding', default='Glove')
    parser.add_argument('--stimes', type=int, help='number of experiment times', default=0)
    parser.add_argument('--times', type=int, help='number of experiment times', default=5)
    parser.add_argument('--experiment', type=int, help='Where to store models and results', default=0)
    opt = parser.parse_args()

    bert_model_path = '../../AnaPython/bert-model/bert-base-uncased/'
    label_schema = 'BIEO'
    # initialize dicts
    dict_list = generate_dict_list(opt.modelname)
    # initialize tokenizer
    if 'Bert' in opt.modelname:
        token_dict, tokenizer = load_bert_tokenizer(bert_model_path)
    else:
        tokenizer = None

    # process train dataset
    train_doc = nerDocSet(opt.datapath + '/train')
    train_doc.load_docset()
    train_doc.preprocess(tokenizer, label_schema)
    train_doc.update_label_dict(dict_list)
    # process test dataset
    test_doc = nerDocSet(opt.datapath + '/test')
    test_doc.load_docset()
    test_doc.preprocess(tokenizer, label_schema)
    test_doc.update_label_dict(dict_list)

    # prepare model input
    if os.path.exists(opt.datapath + '/dev'):
        # process dev dataset if it exists
        valid_doc = nerDocSet(opt.datapath + '/dev')
        valid_doc.load_docset()
        valid_doc.preprocess(tokenizer, label_schema)
        valid_doc.update_label_dict(dict_list)
        # use dev set as validation set
        train_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, False)
        valid_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, False)
        train_doc.X_v, train_doc.y_v = valid_doc.X, valid_doc.y
    else:
        train_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, True)
    test_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, False)

    # train and validate according to options
    for i in range(opt.stimes, opt.times):
        if 't' in opt.option:
            train_doc.train(opt.modelname, dict_list, bert_model_path, 'model{}_{}.hdf5'.format(opt.experiment, i), True, opt.batchsize, opt.epoch, opt.embedding)
        if 'v' in opt.option:
            test_doc.validate(opt.modelname, dict_list, bert_model_path, opt.batchsize, 'model{}_{}.hdf5'.format(opt.experiment, i), 'result{}_{}.txt'.format(opt.experiment, i), opt.embedding)
        if 'a' in opt.option:
            test_doc.test_output('testout{}_{}.txt'.format(opt.experiment, i))
