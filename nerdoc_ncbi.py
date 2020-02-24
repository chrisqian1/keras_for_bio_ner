import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

max_seq_len = 128
max_word_len = 30


class nerDoc():
    def __init__(self, text):
        self.docid = None
        self.text = text
        self.sntlist = []   # list of sentence(only for doc)
        self.tokenlist = [] # list of tokens(only for sent)
        self.emlist = []    # list of entity mention
        self.remlist = []   # list of recognized entity mention

    def split_sentence(self, tokenizer):
        '''
        extract each token and add to tokenlist
        '''

        # split sentence
        for lines in sent_tokenize(self.text):
            sent = nerDoc(lines)
            sent.docid = self.docid
            if tokenizer != None:
                # use word tokenizer
                for word_piece in tokenizer.tokenize(lines):
                    token = nerToken(word_piece)
                    sent.tokenlist.append(token)      
            else:
                for word in word_tokenize(lines):
                    token = nerToken(word)
                    sent.tokenlist.append(token)
            self.sntlist.append(sent)
        return

    def add_POS(self):
        # add pos tags to tokens
        pos_tags = nltk.pos_tag([token.word for token in self.tokenlist])
        for i, pos_tag in enumerate(pos_tags):
            self.tokenlist[i].POS = pos_tag[1]
                
    def align_doc_to_snt(self):
        '''
        align sent tokens position to doc character positions
        '''

        # record the position of each token in original text
        i = 0
        for sent in self.sntlist:
            for token in sent.tokenlist:
                if token.word != '[CLS]' and token.word != '[SEP]':
                    # find the next token in original text
                    while self.text[i].isspace():
                        i+=1
                    spos = i
                    # find the end of the token
                    i = spos + len(token.word.strip('#'))
                    epos = i
                    # record the start and end position of the token
                    token.offset = [spos,epos]
        return
            
    def transfer_entity_from_doc_to_snt(self):
        '''
        transfer entity in doc(with doc character position) to sent(with sent token position)
        '''

        # count entity with index
        entity_i = 0 
        for sent in self.sntlist:
            last_spos = -1
            for token_i, token in enumerate(sent.tokenlist):
                if entity_i < len(self.emlist) and len(token.offset) != 0:
                    entity = self.emlist[entity_i]
                    offset = token.offset
                    # record entity start position
                    if offset[0] == entity.spos:
                        last_spos = token_i
                    # add entity to entity list in sent
                    if offset[1] ==  entity.epos:
                        if last_spos >= 0:
                            sent_entity = EntityMention(entity.entity[0:3]+[last_spos, token_i]+entity.entity[5:])
                            sent.emlist.append(sent_entity)
                        last_spos = -1
                        entity_i += 1
                    # skip if entity does not match the token position
                    if  offset[0] < entity.spos and offset[1] > entity.spos or offset[1] > entity.epos:
                        last_spos = -1
                        entity_i += 1
        return
        
    def assign_labels(self, label_schema):
        '''
        assign labels according to the entity position
        '''

        for em in self.emlist:
            spos, epos = em.spos, em.epos
            for i in range(spos, epos+1):
                # label B or S for single token entity
                if epos == spos:
                    if 'S' in label_schema:
                        tag = 'S' 
                    else:
                        tag = 'B'
                # label B,I,E according to position
                else:
                    if i == spos:
                        tag = 'B'
                    elif i == epos and 'E' in label_schema:
                        tag = 'E'
                    else:
                        tag = 'I'
                self.tokenlist[i].label = '{}-{}'.format(tag, em.type[:3])
        return
        
    def assign_rlabels(self, rlabels, label_dict):
        '''
        assign reconized labels to tokens
        '''
        
        for i,token in enumerate(self.tokenlist):
            for key in label_dict:
                # transfer one hot encoding to original label
                if rlabels[i][label_dict[key]] == np.max(rlabels[i]):
                    token.rlabel = key
        return

    def extract_entity_mention(self):
        '''
        extract gold mention and recognized mention from tokens
        '''

        num = 1
        for sent in self.sntlist:
            sent.remlist = []
            num = sent.recognize_entity_mentions_from_labels(True, num)
        return
        
    def recognize_entity_mentions_from_labels(self, rec, num):
        '''
        add entity to list when there's potential mention
        '''

        last_label, start_pos, entity_content = 'O', -1, []
    
        for i,token in enumerate(self.tokenlist):
            if rec:
                label = token.rlabel
            else:
                label = token.label
            last_type = last_label.split('-')[-1]
            curr_type = label.split('-')[-1]
            tag = label.split('-')[0]

            # add entity when it's the end of an entity
            if last_type != curr_type or tag in 'BS':
                last_label, start_pos, entity_content, num = \
                    self.add_recognized_entity(start_pos, entity_content, last_type, i, rec, num)

            # update last label and entity content
            if label != 'O':
                last_label = label
                entity_content.append(token.word)
        return num
        
    def add_recognized_entity(self, start_pos, entity_content, last_type, pos2, rec, num):
        '''
        add entity to list
        '''
        
        # when it is the end of an entity
        if last_type != 'O':

            # create entity mention content
            while self.tokenlist[start_pos].word[:2] == '##':
                start_pos -= 1
            while self.tokenlist[pos2].word[:2] == '##':
                pos2 += 1
            entity = [self.docid, 'T{}'.format(num), last_type, start_pos, pos2-1, ' '.join(entity_content), '']
            num += 1

            # add to list according to whether it's recognized entity
            if rec:
                self.remlist.append(EntityMention(entity))
            else:
                self.emlist.append(EntityMention(entity))
        # reset variable
        return 'O', pos2, [], num


    def get_entity_position_dict(self):
        '''
        extract gold mention postion and recognized mention position
        '''
        
        gold_dict = dict()
        pred_dict = dict()

        # extract gold mention positions
        for em in self.emlist:
            key = '{}-{}'.format(em.spos, em.epos)
            gold_dict[key] = em.type[:3]
        # extract recognized mention position
        for em in self.remlist:
            key = '{}-{}'.format(em.spos, em.epos)
            pred_dict[key] = em.type
        return gold_dict, pred_dict


class nerToken():
    def __init__(self, word):
        self.word = word
        self.offset = []
        self.POS = ''
        self.label = 'O'
        self.rlabel = ''


class EntityMention():
    def __init__(self, entity):
        self.docid = entity[0]
        self.id = entity[1]
        self.type = entity[2]
        self.spos = int(entity[3])   
        self.epos = int(entity[4])
        self.word = entity[5:-1]
        self.linkid = entity[-1]
        self.entity = entity

