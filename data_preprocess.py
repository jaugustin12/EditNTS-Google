import os
import numpy as np
import pandas as pd
import data
from nltk import pos_tag
from label_edits import sent2edit
from pymetamap import MetaMap
from multiprocessing.pool import Pool
import sys
import time
import nltk
nltk.download('averaged_perceptron_tagger')


# This script contains the reimplementation of the pre-process steps of the dataset
# For the editNTS system to run, the dataset need to be in a pandas DataFrame format
# with columns ['comp_tokens', 'simp_tokens','comp_ids','simp_ids', 'comp_pos_tags', 'comp_pos_ids', edit_labels','new_edit_ids']

PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

def remove_lrb(sent_string):
    # sent_string = sent_string.lower()
    frac_list = sent_string.split('-lrb-')
    clean_list = []
    for phrase in frac_list:
        if '-rrb-' in phrase:
            clean_list.append(phrase.split('-rrb-')[1].strip())
        else:
            clean_list.append(phrase.strip())
    clean_sent_string =' '.join(clean_list)
    return clean_sent_string

def replace_lrb(sent_string):
    sent_string = sent_string.lower()
    # new_sent= sent_string.replace('-lrb-','(').replace('-rrb-',')')
    new_sent = sent_string.replace('-lrb-', '').replace('-rrb-', '')
    return new_sent


def process_raw_data(comp_txt, simp_txt):
    comp_txt = [line.lower().split() for line in comp_txt]
    simp_txt = [line.lower().split() for line in simp_txt]
    # df_comp = pd.read_csv('data/%s_comp.csv'%dataset,  sep='\t')
    # df_simp= pd.read_csv('data/%s_simp.csv'%dataset,  sep='\t')
    assert len(comp_txt) == len(simp_txt)
    df = pd.DataFrame(
                        {'comp_tokens': comp_txt,
                         'simp_tokens': simp_txt,
                        })
    def add_edits(df):
        """
        :param df: a Dataframe at least contains columns of ['comp_tokens', 'simp_tokens']
        :return: df: a df with an extra column of target edit operations
        """
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        pair_sentences = list(zip(comp_sentences,simp_sentences))

        edits_list = [sent2edit(l[0],l[1]) for l in pair_sentences] # transform to edits based on comp_tokens and simp_tokens
        df['edit_labels'] = edits_list
        return df

    def add_pos(df):
        src_sentences = df['comp_tokens'].tolist()
        pos_sentences = [pos_tag(sent) for sent in src_sentences]
        df['comp_pos_tags'] = pos_sentences

        pos_vocab = data.POSvocab('vocab_data/')
        pos_ids_list = []
        for sent in pos_sentences:
            pos_ids = [pos_vocab.w2i[w[1]] if w[1] in pos_vocab.w2i.keys() else pos_vocab.w2i[UNK] for w in sent]
            pos_ids_list.append(pos_ids)
        df['comp_pos_ids'] = pos_ids_list
        return df

    df = add_pos(df)
    df = add_edits(df)
    return df



def get_metamap_op(sents):
    mm_home = '/home/jba5337/work/ds440w/public_mm/bin/metamap20'
    mm = MetaMap.get_instance(mm_home)
    sem_types = ['antb', 'bhvr', 'bmod', 'blor', 'bdsu', 'bdsy', 'chem', 'clna',
             'cnce', 'clnd', 'dsyn', 'enty', 'evnt', 'fndg', 'food', 'ftcn',
             'hlca', 'hlco', 'idcn', 'inch', 'ocdi', 'ocac', 'bpoc', 'orch',
             'podg', 'phsu', 'phpr', 'lbpr', 'resa', 'resd', 'sbst', 'sosy',
             'tmco']
    mm_threshold = 2
    """Replace medical concept mentions with their corresponding UMLS Preferred
    Names"""
    res = []
    for sent in sents:
        sent = sent.strip()
        mm_sent = ''
        concepts, _error = mm.extract_concepts([sent])
        # print(concepts)
        concepts = [c for c in concepts if c[8].count('/') == 1]
        # concepts = [c for c in concepts if any([i in c[5] for i in sem_types])]
        fil_concepts = []
        for concept in concepts:
            try:
                score = float(concept[2])
                if score > mm_threshold:
                    fil_concepts.append(concept)
            except ValueError:
                pass

        # If an identified phrase is mapped to more than one concept, consider
        # the one with the highest score.
        unique_concepts = {}
        for concept in fil_concepts:
            if concept[8] not in unique_concepts:
                unique_concepts[concept[8]] = concept
            else:
                prev_score = unique_concepts[concept[8]][2]
                curr_score = concept[2]
                if curr_score > prev_score:
                    unique_concepts[concept[8]] = concept
        keys = list(unique_concepts.keys())
        sorted_idx = np.argsort([int(key.split('/')[0]) for key in keys])
        ordered_concepts = [unique_concepts[keys[idx]] for idx in sorted_idx]
        start_idx = 0
        for concept in ordered_concepts:
            parts = concept[8].split('/')
            end_idx = int(parts[0]) - 1
            mm_sent += sent[start_idx: end_idx]
            mm_sent += concept[3]
            start_idx = end_idx + int(parts[1])

        # If no concepts are found, copy everything!
        if start_idx != 0:
            mm_sent += sent[start_idx: len(sent)]
        else:
            mm_sent = sent
        res.append(mm_sent)
    return res

def editnet_data_to_editnetID(df,output_path):
    """
    this function reads from df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    and add vocab ids for comp_tokens, simp_tokens, and edit_labels
    :param df: df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids']
    :param output_path: the path to store the df
    :return: a dataframe with df.columns=['comp_tokens', 'simp_tokens', 'edit_labels',
                                            'comp_ids','simp_id','edit_ids',
                                            'comp_pos_tags','comp_pos_ids'])
    """
    out_list = []
    vocab = data.Vocab()
    vocab.add_vocab_from_file('./vocab_data/vocab.txt', 30000)

    def prepare_example(example, vocab):
        """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens', 'edit_labels']
        :param vocab: vocab object for translation
        :return: inp: original input sentence,
        """
        comp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['comp_tokens']])
        simp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['simp_tokens']])
        edit_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['edit_labels']])
        return comp_id, simp_id, edit_id  # add a dimension for batch, batch_size =1

    for i,example in df.iterrows():
        print(i)
        comp_id, simp_id, edit_id = prepare_example(example,vocab)
        ex=[example['comp_tokens'], comp_id,
         example['simp_tokens'], simp_id,
         example['edit_labels'], edit_id,
         example['comp_pos_tags'],example['comp_pos_ids']
         ]
        out_list.append(ex)
    outdf = pd.DataFrame(out_list, columns=['comp_tokens','comp_ids', 'simp_tokens','simp_ids',
                                            'edit_labels','new_edit_ids','comp_pos_tags','comp_pos_ids'])
    outdf.to_pickle(output_path)
    print('saved to %s'%output_path)
    return outdf



comp_simp_ip = ["data/raw/NationalRad/MRI_ABDOMEN_WITH_CONTRAST.en", "data/raw/NationalRad/MRI_ABDOMEN_WITH_CONTRAST.en"]
comp_simp_tgt = ["data/interim/MetaMap/MRI_ABDOMEN_WITH_CONTRAST.en", "data/interim/MetaMap/MRI_ABDOMEN_WITH_CONTRAST.en"]

if __name__ == '__main__':
    f = open("metamap_output.out", 'w')
    start = time.time()
    p = Pool(100)
    for i in range(len(comp_simp_ip)):
        sentences = [l.strip() for l in open(comp_simp_ip[i], 'r').readlines()]
        f.writelines('Document:'+ str(i) + '\n')
        f.writelines([str(sent) + "\n" for sent in sentences])
        ops = get_metamap_op(sentences)
        ops = [str(sent) + "\n" for sent in ops]
        with open(comp_simp_tgt[i], 'w') as tgt_file:
            tgt_file.writelines(ops)
    end = time.time()
    print('METAMAP TOOK:', end - start)
    comp_text = open(comp_simp_tgt[0], "r")
    simp_text = open(comp_simp_tgt[1], "r")
    f.close()
    outdf = editnet_data_to_editnetID(process_raw_data(comp_text, simp_text),'data/interim/MetaMap/MRI_ABDOMEN_WITH_CONTRAST.pkl')
    outdf.to_csv('data/interim/MRI_ABDOMEN_WITH_CONTRAST.csv')
