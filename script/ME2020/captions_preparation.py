#needs pytorch_transformers version 1.2.0
#!/usr/bin/env python
# coding: utf-8

import argparse
import re
import os
import _pickle as cPickle
import numpy as np
import pandas as pd
import torch
from pytorch_transformers.tokenization_bert import BertTokenizer

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)
    
# the same tokenize function from BERT adapted for this task
def tokenize(entries, tokenizer, max_length=16, padding_index=0):
    """Tokenizes the captions.

    This will add c_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_index in embedding
    """
    for entry in entries:
        tokens = tokenizer.encode(entry["caption"])
        tokens = tokens[: max_length - 2]
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)

        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [padding_index] * (max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        assert_eq(len(tokens), max_length)
        entry["c_token"] = tokens
        entry["c_input_mask"] = input_mask
        entry["c_segment_ids"] = segment_ids

# the same tensorize function from BERT adapted for this task
def tensorize(entries, split='trainval'):

    for entry in entries:
        caption = torch.from_numpy(np.array(entry["c_token"]))
        entry["c_token"] = caption

        c_input_mask = torch.from_numpy(np.array(entry["c_input_mask"]))
        entry["c_input_mask"] = c_input_mask

        c_segment_ids = torch.from_numpy(np.array(entry["c_segment_ids"]))
        entry["c_segment_ids"] = c_segment_ids

        if "scores" in entry:
            scores = np.array(entry["scores"], dtype=np.float32)
            scores = torch.from_numpy(scores)
            entry["scores"] = scores

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--captions_path",
        type=str, 
        default="/aloui/MediaEval/dev-set/dev-set_video-captions.txt", 
        help="Captions .txt file"
    )
    
    parser.add_argument(
        "--gt_path",
        type=str, 
        default="/MediaEval/dev-set/ground-truth/ground-truth_dev-set.csv", 
        help="Ground truth .csv file"
    )
    
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        help="which split to use trainval or test"
    )
    
    parser.add_argument(
        "--dc",
        action="store_true",
        help="Whether to use deep captions or not"
    )
    
    args = parser.parse_args()
    
    try:
        assert args.split in ["trainval", "test","train"]
    except Exception as error:
        print("Split must be trainval, train or test")
        raise
    
    #deep_coptions_path = "/MediaEval/alto_titles_danny.csv"
    #train_caption_path = '/aloui/MediaEval/dev-set/dev-set_video-captions.txt'
    dataroot = 'datasets/ME2020'
    max_length = 23
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    entries = []

    if args.dc:
        deep_coptions_df = pd.read_csv(args.captions_path)
        entries = []
        for r in deep_coptions_df.itertuples():
            sample = {}
            vid_id = int(r.video)
            caption = r.caption.rstrip().replace('-', ' ')
            sample['video_id'] = vid_id
            sample['caption'] = caption
            entries.append(sample)
    else:
        df=pd.read_csv(args.captions_path,sep='\t')
        df= df.groupby('video_id').agg({'video_url':'first',
                             'description': ' '.join}).reset_index()
        df['url']=df.video_url.str.split('/').str[-1]
        df['url']=df.url.str.split('.mp4').str[0]

        print(df.description)
        for r in df.itertuples():
          #print(r)
          sample = {}
          #vid_id,video_url, caption = line.split(','
          #vid_id = re.findall(r'\d+', vid_name)[0]
          #caption = caption.rstrip().replace('-', ' ')
          #sample['video_id'] = int(r.video_id)
          sample['video_id'] = r.url
          sample['video_index']=int(r.video_id)
          sample['caption'] = r.description
          entries.append(sample)

    #train_df = pd.read_csv(args.gt_path)
    #score_dict = {}
    #for r in train_df.itertuples():
        #vid_id = r.video_id
        #vid_id = int(vid_id)
        #score_dict[vid_id] = [r.part_1_scores, r.part_2_scores]

    #train_score_list = []
    #for sample in entries:
        #if sample['video_index'] in score_dict:
            #sample['scores'] = score_dict[sample['video_index']]
            #train_score_list.append(sample)



    train_score_list = []
    for sample in entries:
        #if sample['video_index'] in score_dict:
            #sample['scores'] = score_dict[sample['video_index']]
     train_score_list.append(sample)



    tokenize(train_score_list, tokenizer, max_length=max_length)
    tensorize(train_score_list, split=args.split)
    
    #print(len(train_score_list))
    #print(train_score_list[0])
    
    train_cache_path = os.path.join(dataroot, 'cache', 'ME' + '_' + args.split + '_' + str(max_length) + '_cleaned' + '.pkl')
    print("Saving cache file with {} samples under {}".format(len(train_score_list), train_cache_path))
    cPickle.dump(train_score_list, open(train_cache_path, 'wb'))


if __name__ == "__main__":

    main()
