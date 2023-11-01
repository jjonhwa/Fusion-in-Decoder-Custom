import torch
import numpy as np
import random
import pickle

from collections import defaultdict
from typing import List

# Random Seed
def seed_everything(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def read_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def deduplicate_question(query: List, document: List):
    """
        중복되는 query를 최대 2개까지만 채택하며
        이 때, 선택되는 q-d pair는 먼저 등장하는 순서대로 채택한다.

        input: query, document
        return: deduplicated_query, deduplicated_document
    """
    deduplicated_query = []
    deduplicated_document = []

    count_query = defaultdict(int)
    for q, d in zip(query, document):
        count_query[q] += 1
        if count_query[q] <= 2:
            deduplicated_query.append(q)
            deduplicated_document.append(d)
    
    return deduplicated_query, deduplicated_document

def mrr_calculate(candidate_passage_idx):
    """ MRR Score 계산 """
    accumulated_PR = 0
    for idx in range(len(candidate_passage_idx)): # 전체 후보 passage
        for index, context_index in enumerate(candidate_passage_idx[idx]):
            if context_index == idx:
                accumulated_PR += 1 / (1+index)

    MRR = accumulated_PR / (len(candidate_passage_idx))
    return MRR  