import random
from typing import List

import torch
import numpy as np
from intervaltree import IntervalTree


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_overlapping_range(ranges: List[list]) -> torch.tensor:
    """
    Given a list of ranges, remove overlapping ranges.
    Prioritize range with small low end and long range
    """
    if len(ranges) == 0:
        return torch.empty(0, 2, dtype=torch.int64)
    # add 1 so that consecutive ranges are not allowed
    ranges = [[i[0], i[1]+1] for i in ranges]
    ranges = sorted(ranges, key=lambda x: (x[0], x[0]-x[1]))
    tree = IntervalTree()
    tree[ranges[0][0]: ranges[0][1]] = True
    for each in ranges[1:]:
        if not tree.overlaps(*each):
            tree[each[0]: each[1]] = True
    data = list(tree.items())
    data = sorted([[i.begin, i.end-1] for i in data]) # minus 1 to get original ranges
    return torch.tensor(data)


def create_qa_data(texts, labels, ent1s, ent2s, mask_ent=True):
    """
    Create data for QA-based model
    Return: list of sentences, questions, and labels
    """
    templates = [
        "Do {ENT1} and {ENT2} have neutral sentiment toward each other?",
        "Does {ENT1} has positive sentiment toward {ENT2}?",
        "Does {ENT2} has positive sentiment toward {ENT1}?",
        "Does {ENT1} has negative sentiment toward {ENT2}?",
        "Does {ENT2} has negative sentiment toward {ENT1}?"
    ]

    out_text, out_questions, out_label = [], [], []
    for text, ent1, ent2, label in zip(texts, ent1s, ent2s, labels):
        if not mask_ent:
            text = text.replace('[', '')
            text = text.replace(']', '')
        out_text += [text] * len(templates)
        if mask_ent:
            for ques in templates:
                out_questions.append(ques.format(ENT1='<ENT1>', ENT2='<ENT2>'))
        else:
            for ques in templates:
                out_questions.append(ques.format(ENT1=ent1, ENT2=ent2))
        tmp_label = [0] * len(templates)
        tmp_label[label] = 1
        out_label += tmp_label
    assert len(out_text) == len(out_questions)
    assert len(out_text) == len(out_label)
    return out_text, out_questions, out_label
