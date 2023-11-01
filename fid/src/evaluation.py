#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import regex
import string
import unicodedata

from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict
from tqdm import tqdm

import numpy as np

import src.util
"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])

def calculate_matches(data: List, workers_num: int):
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """

    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)

def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for i, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

#################################################
########        READER EVALUATION        ########
#################################################

def _normalize(text):
    return unicodedata.normalize('NFD', text)

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def kor_rouge(prediction, ground_truths, tokenizer, rouge_metric):
    def wsf(text):
        return " ".join(text.split())
    
    def rp(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
        
    prediction = wsf(rp(prediction))

    predictions = tokenizer.encode(prediction)
    references = [tokenizer.encode(gt) for gt in ground_truths]

    rouge_prediction = ' '.join(list(map(str, predictions)))
    rouge_reference = [' '.join(list(map(str, a))) for a in references]

    results = rouge_metric.compute(predictions=[rouge_prediction], references=rouge_reference)

    r1, r2, rL, rLsum = results['rouge1'].mid.fmeasure, results['rouge2'].mid.fmeasure, results['rougeL'].mid.fmeasure, results['rougeLsum'].mid.fmeasure
    return r1, r2, rL, rLsum

def kor_bleu(prediction, ground_truths, tokenizer, bleu_metric):

    def wsf(text):
        return " ".join(text.split())
    
    def rp(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
        
    prediction = wsf(rp(prediction))

    predictions = [tokenizer.tokenize(prediction)]
    references = [tokenizer.tokenize(gt) for gt in ground_truths]

    # 여기 추가함...!
    if predictions == [[]]:
        return 0.0

    results = bleu_metric.compute(predictions=predictions, references=[references])

    return results['bleu']

def kor_f1(prediction, ground_truths, squad_metric):
    
    def wsf(text):
        return " ".join(text.split())
    
    def rp(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
        
    prediction = wsf(rp(prediction))
    predictions = [{'prediction_text': prediction, 'id': 0}]
    references = [{'answers': {'answer_start': [0], 'text': ground_truths}, 'id': 0}]

    results = squad_metric.compute(predictions=predictions, references=references)
    f1, em = results['f1'], results['exact_match']
    return f1, em

def kor_ems(prediction, ground_truths):

    def wsf(text):
        return " ".join(text.split())
    
    def rp(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
        
    prediction = wsf(rp(prediction))

    for gt in ground_truths:
        gt = wsf(rp(gt))
    
        if gt == prediction:
            return 1
    return 0

def get_score(ans_list, gold_list, tokenizer, metric_bleu, metric_rouge, metric_squad, opt):
    
    total = 0
    squad_exactmatch, squad_f1 = [], []
    bleu_score = []
    rouge_1, rouge_2, rouge_L, rouge_Lsum = [], [], [], []

    for ans, gold in tqdm(zip(ans_list, gold_list)):
        BLEU_score = src.evaluation.kor_bleu(ans, gold, tokenizer, metric_bleu)
        ROUGE_1, ROUGE_2, ROUGE_L, ROUGE_Lsum = src.evaluation.kor_rouge(ans, gold, tokenizer, metric_rouge)
        SQUAD_F1, SQUAD_EM = src.evaluation.kor_f1(ans, gold, metric_squad)

        total += 1
                
        squad_exactmatch.append(SQUAD_EM)
        squad_f1.append(SQUAD_F1)
        bleu_score.append(BLEU_score)
        rouge_1.append(ROUGE_1)
        rouge_2.append(ROUGE_2)
        rouge_L.append(ROUGE_L)
        rouge_Lsum.append(ROUGE_Lsum)

    squad_exactmatch, _ = src.util.weighted_average(np.mean(squad_exactmatch), total, opt)
    squad_f1, _ = src.util.weighted_average(np.mean(squad_f1), total, opt)
    bleu_score, _ = src.util.weighted_average(np.mean(bleu_score), total, opt)
    rouge_1, _ = src.util.weighted_average(np.mean(rouge_1), total, opt)
    rouge_2, _ = src.util.weighted_average(np.mean(rouge_2), total, opt)
    rouge_L, _ = src.util.weighted_average(np.mean(rouge_L), total, opt)
    rouge_Lsum, _ = src.util.weighted_average(np.mean(rouge_Lsum), total, opt)
    
    total_score = (squad_exactmatch, squad_f1, bleu_score, rouge_1, rouge_2, rouge_L, rouge_Lsum)
    return total_score
     
####################################################
########        RETRIEVER EVALUATION        ########
####################################################

def eval_batch(scores, inversions, avg_topk, idx_topk):
    for k, s in enumerate(scores):
        s = s.cpu().numpy()
        sorted_idx = np.argsort(-s)
        score(sorted_idx, inversions, avg_topk, idx_topk)

def count_inversions(arr):
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr):
        for j in range(i + 1, lenarr):
            if (arr[i] > arr[j]):
                inv_count += 1
    return inv_count

def score(x, inversions, avg_topk, idx_topk):
    x = np.array(x)
    inversions.append(count_inversions(x))
    for k in avg_topk:
        # ratio of passages in the predicted top-k that are
        # also in the topk given by gold score
        avg_pred_topk = (x[:k]<k).mean()
        avg_topk[k].append(avg_pred_topk)
    for k in idx_topk:
        below_k = (x<k)
        # number of passages required to obtain all passages from gold top-k
        idx_gold_topk = len(x) - np.argmax(below_k[::-1])
        idx_topk[k].append(idx_gold_topk)
