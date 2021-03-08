# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from ._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry

class FeatureExtractionDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        clean_datasets,
        captions_dir,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
        
    ):
        super().__init__()
        self.split = split
        self.num_labels = 2 # hardcoded to test on TASK19
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        clean_train = "_cleaned" if clean_datasets else ""

        cache_path = captions_dir
        
        
        
        if not os.path.exists(cache_path):
            print("Caption path %s does not exist" % cache_path)

        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self, max_length=16):
        """Tokenizes the captions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["c_token"] = tokens
            entry["c_input_mask"] = input_mask
            entry["c_segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self.entries:
            caption = torch.from_numpy(np.array(entry["c_token"]))
            entry["c_token"] = caption

            c_input_mask = torch.from_numpy(np.array(entry["c_input_mask"]))
            entry["c_input_mask"] = c_input_mask

            c_segment_ids = torch.from_numpy(np.array(entry["c_segment_ids"]))
            entry["c_segment_ids"] = c_segment_ids

            if "test" not in self.split:
                # answer = entry["answer"]
                # labels = np.array(answer["labels"])
                scores = np.array(entry["scores"], dtype=np.float32)
                
                scores = torch.from_numpy(scores)
                entry["scores"] = scores
                # entry["answer"]["scores"] = scores

                '''if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None'''

    def __getitem__(self, index):
        entry = self.entries[index]
        video_id = entry["video_id"]
        print(video_id)
        # question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[video_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        # shuffle the image location here.
        # img_idx = list(np.random.permutation(num_boxes-1)[:mix_num_boxes]+1)
        # img_idx.append(0)
        # mix_boxes_pad[:mix_num_boxes] = boxes[img_idx]
        # mix_features_pad[:mix_num_boxes] = features[img_idx]

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        caption = entry["c_token"]
        input_mask = entry["c_input_mask"]
        segment_ids = entry["c_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        
        target = torch.zeros(self.num_labels)
        if "scores" in entry:
            target = entry["scores"]
        
        '''if "test" not in self.split:
            # answer = entry["answer"]
            # labels = answer["labels"]
            target = entry["scores"]
            # if labels is not None:
            #     target.scatter_(0, labels, scores)'''
        
        return (
            features,
            spatials,
            image_mask,
            caption,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            # question_id,
        )

    def __len__(self):
        return len(self.entries)
