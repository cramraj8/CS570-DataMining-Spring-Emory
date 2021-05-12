import torch
from bunch import Bunch
from torch.utils.data import Dataset
import os
import json
import glob
import pandas as pd
from transformers import BertTokenizer
from collections import defaultdict


def map_to_torch_float(encoding):
    encoding = torch.FloatTensor(encoding)
    encoding.requires_grad_(False)
    return encoding


def map_to_torch(encoding):
    encoding = torch.LongTensor(encoding)
    encoding.requires_grad_(False)
    return encoding


def encode_train_sequence(qText, posDocText, negDocText,
                          max_seq_len, max_seq_query_len, tokenizer):
    seqA = tokenizer.tokenize(qText)[:max_seq_query_len]
    seqA_len = len(seqA)
    seqA = ["[CLS]"] + seqA + ["[SEP]"]

    # =============================================================================
    posSeqB = tokenizer.tokenize(posDocText)[:max_seq_len - len(seqA) - 3]
    posSeqB_len = len(posSeqB)
    posSeqB = posSeqB + ["[SEP]"]
    pos_input_tokens = seqA + posSeqB
    pos_input_ids = tokenizer.convert_tokens_to_ids(pos_input_tokens)
    pos_input_mask = [1] * len(pos_input_ids)
    pos_sequence_ids = [0] * len(seqA) + [1] * len(posSeqB)  # TODO: How about filling [seq] and [cls] with IDs ???
    while len(pos_input_ids) < max_seq_len:
        pos_input_ids.append(0)
        pos_input_mask.append(0)
        pos_sequence_ids.append(0)  # TODO: is this correct? or we pad seq_ID with some other ?

    # =============================================================================
    negSeqB = tokenizer.tokenize(negDocText)[:max_seq_len - len(seqA) - 3]
    negSeqB_len = len(negSeqB)
    negSeqB = negSeqB + ["[SEP]"]
    neg_input_tokens = seqA + negSeqB
    neg_input_ids = tokenizer.convert_tokens_to_ids(neg_input_tokens)
    neg_input_mask = [1] * len(neg_input_ids)
    neg_sequence_ids = [0] * len(seqA) + [1] * len(negSeqB)
    while len(neg_input_ids) < max_seq_len:
        neg_input_ids.append(0)
        neg_input_mask.append(0)
        neg_sequence_ids.append(0)  # TODO: is this correct? or we pad seq_ID with some other ?

    return map_to_torch(pos_input_ids), map_to_torch(pos_input_mask), map_to_torch(pos_sequence_ids), \
           map_to_torch(neg_input_ids), map_to_torch(neg_input_mask), map_to_torch(neg_sequence_ids), \
           seqA_len, posSeqB_len, negSeqB_len


def encode_test_sequence(qText, passageText,
                          max_seq_len, max_seq_query_len, tokenizer):
    seqA = tokenizer.tokenize(qText)[:max_seq_query_len]
    seqA_len = len(seqA)
    seqA = ["[CLS]"] + seqA + ["[SEP]"]

    # =======================================================
    seqB = tokenizer.tokenize(passageText)[:max_seq_len - len(seqA) - 3]
    seqB_len = len(seqB)
    seqB = seqB + ["[SEP]"]

    input_tokens = seqA + seqB

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    sequence_ids = [0] * len(seqA) + [1] * len(seqB)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        sequence_ids.append(0)  # TODO: is this correct? or we pad seq_ID with some other ?

    return map_to_torch(input_ids), map_to_torch(input_mask), map_to_torch(sequence_ids), seqA_len, seqB_len


class TrainTRECDataset(Dataset):
    def __init__(self, file_name, conf, bert_query_max_len=20, bert_max_len=512, is_train=True, bert_tokenizer=None):
        self.config_model = conf
        self.is_train = is_train
        self.bert_tokenizer = bert_tokenizer
        self.bert_max_len = self.config_model['data']['max_seq_len']
        self.bert_query_max_len = self.config_model['data']['max_query_len']

        self.qLookup = defaultdict()
        self.dLookup = defaultdict()
        self.data_df = self.load_data(file_name)
        self.len = len(self.data_df)

    def load_data(self, data_path):
        data = []
        if os.path.isdir(data_path):
            train_files = sorted(glob.glob(os.path.join(data_path, "fold-*.json")))
            for train_file in train_files:
                tmp_data = json.load(open(train_file, 'r'))
                data.extend(tmp_data)
        else:
            data = json.load(open(data_path, 'r'))

        return_data = {'qID': [], 'qText': [], 'dID': [], 'dText': [], 'label': []}
        for data_sample in data:
            qID = data_sample['qID']
            # q_ent_tokens_text = ' '.join([e['entity_title'] for e in data_sample['qEntities']])
            # if q_ent_tokens_text == '': continue
            # self.qLookup[qID] = q_ent_tokens_text
            
            n_doc = 0
            for rel_docs in data_sample['RelevantDocuments']:
                dID = rel_docs['docID']
                # d_ent_tokens_text = ' '.join([e['entity_title'] for e in rel_docs['dEntities']])
                # if d_ent_tokens_text == '': continue
                # self.dLookup[dID] = d_ent_tokens_text

                tmp_doc_text = rel_docs['docText'].split("\t")
                if len(tmp_doc_text) < 5: continue  # TODO: 0 or 5
                n_doc += 1

                self.dLookup[dID] = rel_docs['docText']

                return_data['qID'].append(qID)
                # return_data['qText'].append( q_ent_tokens_text )
                return_data['dID'].append(dID)
                # return_data['dText'].append( d_ent_tokens_text )
                # return_data['label'].append(1)

            if n_doc > 0:
                self.qLookup[qID] = data_sample['qString']

        # df = pd.DataFrame(return_data, columns=["qID", "qText", "dID", "dText", "label"])
        df = pd.DataFrame(return_data, columns=["qID", "dID"])
        return df

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len
        inst = self.data_df.loc[index]
        docID = inst['dID']
        qID = inst['qID']
        docText = self.dLookup[docID]
        qText = self.qLookup[qID]

        negInst = self.data_df[self.data_df['qID'] != qID].sample(n=3, replace=True, random_state=1).iloc[0]
        # negInst = self.data_df[self.data_df['qID'] != qID].iloc[0] # get a random element
        negDocID = negInst['dID']
        negDocText = self.dLookup[negDocID]

        # print(qID)
        # print(qText)
        # print()
        # print(docID)
        # print(docText)
        # print()
        # print(negDocID)
        # print(negDocText)

        pos_ids, pos_mask, pos_type_ids, \
        neg_ids, neg_mask, neg_type_ids, \
        seqA_len, posSeqB_len, negSeqB_len = encode_train_sequence(qText, docText, negDocText,
                                                                   self.bert_max_len, self.bert_query_max_len,
                                                                   self.bert_tokenizer)

        label = 1
        return pos_ids, pos_mask, pos_type_ids, neg_ids, neg_mask, neg_type_ids, seqA_len, posSeqB_len, negSeqB_len, map_to_torch_float(
            [label])


class TestTRECDataset(Dataset):
    def __init__(self, file_name, conf, bert_query_max_len=20, bert_max_len=512, is_train=False, bert_tokenizer=None):
        self.config_model = conf
        self.is_train = is_train
        self.bert_tokenizer = bert_tokenizer
        self.bert_max_len = self.config_model['data']['max_seq_len']
        self.bert_query_max_len = self.config_model['data']['max_query_len']

        self.qLookup = defaultdict()
        self.dLookup = defaultdict()
        self.data_df = self.load_data(file_name)
        self.len = len(self.data_df)

    def load_data(self, data_path):
        data = []
        if os.path.isdir(data_path):
            train_files = sorted(glob.glob(os.path.join(data_path, "fold-*.json")))
            for train_file in train_files:
                tmp_data = json.load(open(train_file, 'r'))
                data.extend(tmp_data)
        else:
            data = json.load(open(data_path, 'r'))

        return_data = {'qID': [], 'qText': [], 'dID': [], 'dText': [], 'label': []}
        for data_sample in data:
            qID = data_sample['qID']
            # q_ent_tokens_text = ' '.join([e['entity_title'] for e in data_sample['qEntities']])
            # self.qLookup[qID] = q_ent_tokens_text
            self.qLookup[qID] = data_sample['qString']

            for rel_docs in data_sample['RelevantDocuments']:
                dID = rel_docs['docID']
                # d_ent_tokens_text = ' '.join([e['entity_title'] for e in rel_docs['dEntities']])
                # self.dLookup[dID] = d_ent_tokens_text
                self.dLookup[dID] = rel_docs['docText']

                return_data['qID'].append(qID)
                # return_data['qText'].append( q_ent_tokens_text )
                return_data['dID'].append(dID)
                # return_data['dText'].append( d_ent_tokens_text )
                # return_data['label'].append(1)

        # df = pd.DataFrame(return_data, columns=["qID", "qText", "dID", "dText", "label"])
        df = pd.DataFrame(return_data, columns=["qID", "dID"])
        return df

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len
        inst = self.data_df.loc[index]
        docID = inst['dID']
        qID = inst['qID']
        docText = self.dLookup[docID]
        qText = self.qLookup[qID]

        # print(qID)
        # print(qText)
        # print()
        # print(docID)
        # print(docText)

        ids, mask, type_ids, seqA_len, seqB_len = encode_test_sequence(qText, docText,
                                                                       self.bert_max_len, self.bert_query_max_len,
                                                                       self.bert_tokenizer)

        label = 1
        return [ids, mask, type_ids, map_to_torch_float([label]), qID, docID, seqA_len, seqB_len]


if __name__ == "__main__":
    def get_config_from_json(json_file):
        """
        Get the config from a json file
        :param json_file:
        :return: config(namespace) or config(dictionary)
        """
        # parse the configurations from the config json file provided
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)

        # convert the dictionary to a namespace using bunch lib
        config = Bunch(config_dict)

        return config, config_dict


    BENCHMARK_TRAIN_DIR = "../Entity-Linking/Train-with-entities/" # "../Data/Train-with-entities/"
    BENCHMARK_TEST_FILE = "../Entity-Linking/Test-with-entities/ramraj-test-data-top100-BM25-opt.json" # "../Data/Test-with-entities/ramraj-test-data-top100-BM25-opt.json"

    config, _ = get_config_from_json("config.json")
    bert_tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"],
                                                   cache_dir=config.data['pretrained_download_dir'])
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # train_trec_dataset = TrainTRECDataset(BENCHMARK_TRAIN_DIR, config, 	is_train=True, bert_tokenizer=bert_tokenizer)
    # element1 = train_trec_dataset.__getitem__(0)

    test_trec_dataset = TestTRECDataset(BENCHMARK_TEST_FILE, config, is_train=False, bert_tokenizer=bert_tokenizer)
    element1 = test_trec_dataset.__getitem__(0)

