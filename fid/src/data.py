# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np

# Original FiD Datset
class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context = None, # 질문당 후보 Context의 개수
                 question_prefix = 'question:',
                 passage_prefix = 'context:' # title은 제외
                 ):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)
    
    def get_target(self, example):
        if 'summary' in example:
            target = example['summary']
            return target + ' </s>'
        if 'target' in example: # 정답이 하나 존재할 경우
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example: # 정답이 여러 개 존재할 경우
            return random.choice(example['answers']) + " </s>" # 아무거나 1개로 학습되도록 함
        else:
            return None
        
    def __getitem__(self, index):
        example = self.data[index]

        # "question: ..."
        if 'summary' not in example:
            question = self.question_prefix + " " + example['question'] 
        
        target = self.get_target(example) # " </s>" 삽입

        if 'ctxs' in example and self.n_context is not None:

            if 'summary' in example:
                contexts = example['ctxs'][:self.n_context]
                passages = [c['text'] for c in contexts]

                return {
                    'index': index,
                    'target': target,
                    'passages': passages,
                }
            
            else:
                f = self.passage_prefix + " {}"
                contexts = example['ctxs'][:self.n_context] # context의 개수를 여기서 조정할 수 있음
                passages = [f.format(c['text']) for c in contexts] # "[context: ...]"의 형태로 만듬
                # score 생략함
                
                if len(contexts) == 0: # context가 없을 경우 question으로 대체 ( 왜지 ? )
                    contexts = [question]
        
        else:
            passages = None

        return {
            'index': index,
            'question': question,
            'target': target,
            'passages': passages
        }
    
    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        
        for ex in self.data:
            ex['ctxs'].sort(key = lambda x: float(x['score']), reverse=True)
    
    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )

        # List에서 [None]을 주면 1차원이 맨 앞쪽에 생김 => Batch를 생성해주기 위함
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    # dim=0을 기준으로 묶어줌으로써 Batch를 생성함
    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer_input, tokenizer_output=None, decoder_maxlength=None, answer_maxlength=20):
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output
        self.text_maxlength = text_maxlength
        self.decoder_maxlength = decoder_maxlength
        self.answer_maxlength = answer_maxlength


    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]

        # batch_encode_plus => Tokenizing을 하는 데 batch 기준 (즉, list 차원이 하나 더 생김 )으로 진행
        if self.tokenizer_output:
            target = self.tokenizer_output.batch_encode_plus(
                target,
                # max_length = None ===> 가장 긴 녀석에 맞춰짐
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                pad_to_max_length=True, 
                return_tensors='pt',
                truncation=True if self.answer_maxlength > 0 else False,
            )

            # Decoder Model의 Decoder Max Length로 맞춰줌
            seq_length = target['input_ids'].size(1)
            if seq_length > self.decoder_maxlength:
                target['input_ids'] = target['input_ids'][:, :self.decoder_maxlength]
                target['attention_mask'] = target['attention_mask'][:, :self.decoder_maxlength]

                batch_size = target['input_ids'].size(0)
                for batch_idx in range(batch_size):
                    if target['input_ids'][batch_idx][-1] != self.tokenizer_output.pad_token_id:
                        target['input_ids'][batch_idx][-1] = 1
        else:
            target = self.tokenizer_input.batch_encode_plus(
                target,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                pad_to_max_length=True, 
                return_tensors='pt',
                truncation=True if self.answer_maxlength > 0 else False,
            )

        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()

        # mask=False인 부분을 -100으로 masking
        target_ids = target_ids.masked_fill(~target_mask, -100) 

        # "['question: ... title: ... context: ...']"의 형태로 만듦        
        def append_question(example):
            if 'question' not in example:
                return example['passages']
            
            if example['passages'] is None:
                return [example['question']]
            
            # example['passages']가 여러 개 있을 수 있음
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer_input,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    
    
    examples = []
    
    # data: List => example: 1개의 question에 대한 정보 ( 정답, 100개의 Passage )
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example: # id 만들기
            example['id'] = k

        # 잘못된거 같은데 score를 어차피 사용하지 않음
        for c in example['ctxs']: # score가 없을 경우 passage 순서대로 점수를 부여하는 듯 해보임 ( 1/2, 1/3, 1/4, 1/5, ... )
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)