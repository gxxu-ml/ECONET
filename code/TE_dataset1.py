import json
import re
import pandas as pd
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer as twt
import spacy
from collections import OrderedDict
from tqdm import tqdm
import pickle




#path of three files, needed to run te_dataset creation
srl_path = '../bamboo-cutter-moon-child.srl.json'
sentences_path = '../bamboo-cutter-moon-child.sentences.csv'
coref_args_path = "../bamboo_coref_args_df.csv"

with open(srl_path, 'r') as json_file:
    srl = json.load(json_file)
sentences_df = pd.read_csv(sentences_path)
sentences = sentences_df['text'].tolist()
coref_args_df = pd.read_csv(coref_args_path)



nlp = spacy.load("en_core_web_sm")
def spacy_span(sent):
    """
    get a list of spans using spacy tokenizer
    """
    spans = []
#     sent = sentences[sent_id]
    doc = nlp(sent)
    token_id = 0
    for offset in range(len(sent)):
        if token_id >= len(doc):
            continue
        token=doc[token_id]

        if sent[offset: offset+len(token)] == token.text:
            spans.append((offset, offset+len(token)))
            token_id+=1
#             print(sent[offset: offset+len(token)],token.text)
    return spans

def spacy_tokenize(sent):
    """
    output a list of tokens, on par with ntlk word_tokenize
    """
    #     sent = sentences[sent_id]
    doc = nlp(sent)
    out = []
    for token in doc:
        out.append(token.text)
    return out



def byteParse(sentence):
    """
    parse a sentence into ordered dict, needed by TE data format
    """
    pos_tags_sentence = OrderedDict()
#     pos_tags = pos_tag(word_tokenize(sentence), tagset='universal')
    pos_tags = pos_tag(spacy_tokenize(sentence), tagset='universal')

#     token_spans = list(twt().span_tokenize(sentence))
    token_spans = spacy_span(sentence)
    if len(pos_tags) == len(token_spans):
        for i, span in enumerate(token_spans):
            pos_tags_sentence['['+str(span[0])+':'+ str(span[1]) + ')'] = (pos_tags[i][0], pos_tags[i][1])
#             pos_tags_sentence[(span[0], span[1])] = (pos_tags[i][0], pos_tags[i][1])
    else:
        for i, span in enumerate(token_spans):
            if sentence[span[0]:span[1]] ==  pos_tags[i][0]:
                pos_tags_sentence['['+str(span[0])+':'+ str(span[1]) + ')'] = (pos_tags[i][0], pos_tags[i][1])
            else:
                pos_tags_sentence['['+str(span[0])+':'+ str(span[1]) + ')'] = (pos_tags[i+1][0], pos_tags[i+1][1])

    return pos_tags_sentence

def event_span(sent_id, verb_id, srl=srl, sentences=sentences):
    """
    need srl, sentences as default preset global variables
    """
    idx = srl[sent_id]['verbs'][verb_id]['tags'].index('B-V')
#     sent_token_span = list(twt().span_tokenize(sentences[sent_id]))
    sent_token_span = spacy_span(sentences[sent_id])
    spans = (sent_token_span[idx][0],sent_token_span[idx][1]-1)
    return spans

# this is the event class defined by the ECONET Event tool;
class Event():
    def __init__(self, id, type, text, tense, polarity, span):
        self.id = id
        self.type = type
        self.text = text
        self.tense = tense
        self.polarity = polarity
        self.span = span


def create_data_instance(sent_id1, sent_id2, verb_id1, verb_id2, sentences=sentences, srl = srl):
    """
    sent_id1, and sent_id2 are two sentences we want to run
    verb_id1, is the verb_id in sent1, verb_id2, is the verb_id in sentence2

    ####modify it for situation when two events are in the same sentence!!!!!!
    """
    print("currently processing the following event pair")
    print(sent_id1, sent_id2, verb_id1, verb_id2)
    if sent_id1 != sent_id2:
        print("offset choice")
        combo_sent = sentences[sent_id1]+ ' ' + sentences[sent_id2]
        doc_dict = byteParse(combo_sent)

        e1_span = event_span(sent_id1, verb_id1)
        ori_e2 = event_span(sent_id2, verb_id2)
        offset = len(sentences[sent_id1])+1
        e2_span =  (ori_e2[0] + offset, ori_e2[1]+offset)
    else:
        combo_sent = sentences[sent_id1]
        doc_dict = byteParse(combo_sent)

        e1_span = event_span(sent_id1, verb_id1)
        e2_span = event_span(sent_id2, verb_id2)


    # if combo_sent[e2_span[0]: e2_span[1]+1] != srl[sent_id2]['verbs'][verb_id2]['verb']:
    #     print(combo_sent[e2_span[0]: e2_span[1]+1], srl[sent_id2]['verbs'][verb_id2]['verb'])
    #     print(e2_span)
    #     print(sent_id2)

    assert combo_sent[e1_span[0]: e1_span[1]+1] == srl[sent_id1]['verbs'][verb_id1]['verb']
    assert combo_sent[e2_span[0]: e2_span[1]+1] == srl[sent_id2]['verbs'][verb_id2]['verb']

    #below stay the same
    left_event = Event(None, None, None, None, None, e1_span)
    right_event = Event(None, None, None, None, None, e2_span)
    v = dict()
    v['rel_type'],v['rev'],v['doc_dictionary'],v['event_labels'],v['doc_id'],v['left_event'],v['right_event'] = \
    'BEFORE', None, doc_dict, None, None, left_event, right_event
    return v



def get_auxis(coref_args_df):
    """
    Outputs new column indicating whether it's an auxiliary verb
    """
    aux_ls = ['is', 'are','was','were','be','been','has','have','had','going','can','could','will','would','shall','should',
             'may','might','must','do','does','did']
    aux = []
    for i in range(coref_args_df.shape[0]):
        line = coref_args_df.iloc[i]
        verb = line['verb']
        if verb in aux_ls:
            aux.append(True)
        else:
            aux.append(False)
    return aux

def get_major_event(filter_df, sentence_id):
    """
    output the verb_id of the sentence, that is the most preferred;
    two heuristicd used:
    1. auxiliary verb is not preferred
    2. more arguments are preferred
    """
    tmp = filter_df[filter_df['sentence_id']==sentence_id]
    d = dict()
    if tmp.shape[0]==0:
        #when there's no verb for this sentence
        return None
    for i in range(tmp.shape[0]):
        if tmp.iloc[i]['auxi']== True:
            d[tmp.iloc[i]['verb_id']] = tmp.iloc[i]['args_n']-10
        else:
            d[tmp.iloc[i]['verb_id']] = tmp.iloc[i]['args_n']

    return max(d, key=d.get)


def create_data_instances(coref_args_df):
    """
    Main function to create data create_data_instances that can be stored in pickle files
    Extract one event per sentence
    """
    coref_sent_ids = list(set(coref_args_df['sentence_id']))

    d = dict()
    for i in tqdm(range(len(coref_sent_ids)-1)):
        s_id = coref_sent_ids[i]
        e_id = get_major_event(coref_args_df, s_id)
        s_id2 = coref_sent_ids[i+1]
        e_id2 = get_major_event(coref_args_df, s_id2)
#         print(s_id,s_id2,e_id,e_id2)
        v = create_data_instance(s_id,s_id2,e_id,e_id2)
        d['L_' + str(i)] = v
    return d


def create_TE_ids(coref_args_df):
    """
    Create the TE_ids data for the main data creation function;
    """
    #     coref_sent_ids = list(set(coref_args_df['sentence_id']))
    d = dict()
    # get rid of all auxilliary verbs;
    coref_args_df = coref_args_df[coref_args_df['auxi']==False]
    for i in range(coref_args_df.shape[0]):
        row = coref_args_df.iloc[i]
        s_id = row['sentence_id']
        if s_id not in d:
            d[s_id] = set()
        d[s_id].add(row['verb_id'])
    prev = None

    TE_ids = []
    print("the length of dictionary", len(d.keys()))
    for s_id in d.keys():
        verb_ids = list(d[s_id])
        for v_id in verb_ids:
            if prev == None:
                prev = [s_id, v_id]
                continue
            TE_ids.append( [prev[0], prev[1], s_id, v_id] )
            prev = [s_id, v_id]
    return TE_ids







def create_data_instances_multi_events(TE_ids):
    """
    Main function to create data create_data_instances that can be stored in pickle files
    It takes a TE_ids file, eg:  list [ list [s_id,s_id2,e_id,e_id2],  [s_id',s_id2',e_id',e_id2'], ...    ]
    Rather than extracting one event per sentence, it extracts all eligible events, and do temperal ordering of them;
    """
    d = dict()
    for i in tqdm(range(len(TE_ids))):
        s_id,e_id,s_id2,e_id2 = TE_ids[i]
        v = create_data_instance(s_id,s_id2,e_id,e_id2)
        d['L_' + str(i)] = v
    return d







if __name__ == "__main__":
    coref_args_df['auxi'] = get_auxis(coref_args_df)
    #
    # data = create_data_instances(coref_args_df)
    #
    print('the shape of coref_args_df is: ', coref_args_df.shape)
    TE_ids = create_TE_ids(coref_args_df)
    print('the length of TE_ids is: ', len(TE_ids))
    data = create_data_instances_multi_events(TE_ids)

    with open('../data/matres/bamboo_example3.pickle', 'wb') as f:
        pickle.dump(data, f)


