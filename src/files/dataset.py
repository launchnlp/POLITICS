import os
import json
import re
import pickle
import random
from typing import List, Dict, Tuple, Optional

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from src.files.utils import create_qa_data, remove_overlapping_range
from src.util import config


LEAK_INFO = [
    r"daily\s?kos",
    r"(the\s?)?huff(ington)?\s?post",
    r"\bcnn\b",
    r"(the\s?)?washington\s?post",
    r"(the\s?)?n(ew\s?)?y(ork\s?)?times",
    r"usatoday",
    r"the\s?hill",
    r"\bap(news)?\b", 
    r"fox\s?news",
    r"breitbart",
    r"washington\s?times"
]

ALL_LEAK_INFO = [
    r"daily\s?kos",
    r"(the\s?)?huff(ington)?\s?post",
    r"\bcnn\b",
    r"(the\s?)?washington\s?post",
    r"(the\s?)?n(ew\s?)?y(ork\s?)?times",
    r"usatoday",
    r"the\s?hill",
    r"\bap(news)?\b", 
    r"fox\s?news",
    r"breitbart",
    r"washington\s?times",
    r"town\s?hall",
    r"news\s?max",
    r"\bbbc\b",
    r"reuters",
    r"christian science monitor",
    r"national\s?review",
    r"washington\s?examiner",
    r"\bnpr\b",
    r"\bpolitico\b",
    r"the\s?guardian",
    r"\bvox\b",
    r"the\s?blaze",
    r"wall street journal",
    r"\bcbn\b"
]


ENTITY_TYPES = {'PERSON', 'NORP', 'ORG', 'GPE', 'EVENT'}


def load_cache_file(cache_file: str, data_file: Optional[str]=None) -> \
        Tuple[list, bool, Optional[list]]:
    """Check if cache file exists, if yes, load it, if not, return false"""
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as fp:
            cache = pickle.load(fp)
        return cache, True, None
    else:
        data = None
        if data_file is not None:
            with open(data_file, 'rb') as fp:
                data = pickle.load(fp)
        return [], False, data


def store_cache(cache_file: str, data, loaded: bool):
    """If the file is not loaded, store to cache_file"""
    if not loaded:
        with open(cache_file, 'wb') as fp:
            pickle.dump(data, fp)


class MLMDataset(Dataset):
    """Pretraining Dataset for MLM objective."""

    def __init__(self, files: List[str], model_name: str, mask_entity: bool=False,
            mask_sentiment: bool=False, lexicon_dir: str=''):
        super().__init__()
        batch_size = 128
        self.entity_index, self.sentiment_index = [], []
        self.input_ids, self.attentions = [], []
        self.labels = []
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_space = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        if mask_sentiment:
            sentiment_lexicon = list(load_lexicon(lexicon_dir))
            # separate word length, only include <= 5 tokens
            sentiment_ids = {1: [], 2: [], 3: [], 4: [], 5: []}
            for word in sentiment_lexicon:
                ids_space = tokenizer_space(word, return_tensors='pt',
                    add_special_tokens=False)['input_ids']
                if ids_space.shape[1] > 5:
                    continue
                sentiment_ids[ids_space.shape[1]].append(ids_space)
            sentiment_ids = {key: torch.vstack(value) for key, value in sentiment_ids.items()}
        for file_ind, file_i in enumerate(files):
            with open(file_i, 'r') as fp:
                data = json.load(fp)
                data_num = len(data)
            basename, _ = os.path.splitext(file_i)
            entity_index_file = f"{basename}_entity_index.pkl"
            entity_file = f"{basename}_entity.pkl"
            sentiment_index_file = f"{basename}_sentiment_index.pkl"
            inputs_file = f"{basename}_inputs.pkl"

            # load cached data
            tmp, loaded_inputs, _ = load_cache_file(inputs_file)
            if len(tmp) == 0:
                file_input_ids, file_attentions = [], []
            else:
                file_input_ids, file_attentions = tmp
            # ideology labels
            self.labels += [file_ind] * data_num
            if mask_entity:
                file_entity_index, loaded_entity, entity = load_cache_file(entity_index_file, entity_file)
                if not loaded_entity:
                    assert len(entity) == data_num
            if mask_sentiment:
                file_sentiment_index, loaded_sentiment, _ = load_cache_file(sentiment_index_file)
                if loaded_sentiment:
                    assert len(file_sentiment_index) == data_num
            
            for ind in tqdm(range(0, data_num, batch_size)):
                if not loaded_inputs:
                    batch = data[ind: ind+batch_size]
                    text = [i['title'] + ' \n' + ' '.join(i['text']) for i in batch]
                    # mask leaking media info
                    for pattern in LEAK_INFO:
                        text = [re.sub(pattern, '<mask>', i, flags=re.IGNORECASE) for i in text]
                    # get inputs
                    inputs = tokenizer(text, padding='max_length', truncation=True,
                        return_tensors="pt")
                    file_input_ids.append(inputs['input_ids'])
                    file_attentions.append(inputs['attention_mask'])
                # get position of named entities
                if mask_entity and not loaded_entity:
                    batch_entity = entity[ind: ind+batch_size]
                    for id_in_batch, each in enumerate(batch_entity):
                        file_entity_index.append(get_article_entity_index(each,
                            file_input_ids[ind//batch_size][id_in_batch], tokenizer, tokenizer_space))
                # get position of sentiment words
                if mask_sentiment and not loaded_sentiment:
                    for id_in_batch in range(file_input_ids[ind//batch_size].shape[0]):
                        file_sentiment_index.append(get_article_sentiment_index(sentiment_ids,
                            file_input_ids[ind//batch_size][id_in_batch]))
            if mask_entity:
                assert len(file_entity_index) == data_num
                self.entity_index += file_entity_index
                store_cache(entity_index_file, file_entity_index, loaded_entity)
            if mask_sentiment:
                assert len(file_sentiment_index) == data_num
                self.sentiment_index += file_sentiment_index
                store_cache(sentiment_index_file, file_sentiment_index, loaded_sentiment)
            store_cache(inputs_file, (file_input_ids, file_attentions), loaded_inputs)
            self.input_ids += file_input_ids
            self.attentions += file_attentions
        
        self.input_ids = torch.cat(self.input_ids, 0)
        self.attentions = torch.cat(self.attentions, 0)
        if mask_entity:
            self.entity_index = torch.nn.utils.rnn.pad_sequence(self.entity_index,
                batch_first=True, padding_value=-1)
        if mask_sentiment:
            self.sentiment_index = torch.nn.utils.rnn.pad_sequence(self.sentiment_index,
                batch_first=True, padding_value=-1)
        self.mask_entity = mask_entity
        self.mask_sentiment = mask_sentiment
        assert len(self.labels) == self.input_ids.shape[0]
        
    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        special_index = []
        if self.mask_entity:
            special_index.append(self.entity_index[idx][self.entity_index[idx, :, 0] != -1])
        if self.mask_sentiment:
            special_index.append(self.sentiment_index[idx][self.sentiment_index[idx, :, 0] != -1])
        if self.mask_entity or self.mask_sentiment:
            return self.input_ids[idx], self.attentions[idx], self.labels[idx], special_index
        else:
            return self.input_ids[idx], self.attentions[idx], self.labels[idx]


class TupleDataset(Dataset):
    """Pretraining Dataset for contrastive loss objectives."""

    def __init__(self, filename: str, model_name: str):
        super().__init__()
        self.source_idx, self.ideo_idx = [], []
        with open(filename, 'r') as fp:
            data = json.load(fp)
        basename, _ = os.path.splitext(filename)
        inputs_file = f"{basename}_inputs.pkl"
        inputs, loaded, _ = load_cache_file(inputs_file)
        if loaded:
            self.input_ids, self.attentions, self.tuples = inputs
        else:
            self.input_ids, self.attentions = [], [] # stores inputs for the unique set of articles
            self.tuples = [] # N * 11 tensor, store index to self.input_ids and self.attentions
            article_to_idx = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        for each in tqdm(data):
            # get source indices
            source_idx = [config.SOURCE_to_IND[i['source']] for i in each]
            self.source_idx.append(torch.tensor(source_idx))
            ideo_idx = [config.SOURCE_to_IDEO[i['source']] for i in each]
            self.ideo_idx.append(torch.tensor(ideo_idx))
            # check
            assert self.ideo_idx[-1].shape == self.source_idx[-1].shape
            if not loaded:
                # get un-encoded articles
                unencoded = [i for i in each if (i['source'], i['date'], i['title']) not in article_to_idx]
                if len(unencoded) > 0:
                    # concat title and text
                    text = [i['title'] + '\n' + ' '.join(i['text']) for i in unencoded]
                    # mask leaking media info
                    for pattern in LEAK_INFO:
                        text = [re.sub(pattern, '<mask>', i, flags=re.IGNORECASE) for i in text]
                    # encode
                    inputs = self.tokenizer(text, padding='max_length', truncation=True,
                        return_tensors="pt")
                    # add to inputs
                    for ind in range(len(unencoded)):
                        article_to_idx[(unencoded[ind]['source'], unencoded[ind]['date'],
                            unencoded[ind]['title'])] = len(article_to_idx)
                        self.input_ids.append(inputs['input_ids'][ind])
                        self.attentions.append(inputs['attention_mask'][ind])
                # map tuple to actual inputs
                self.tuples.append(torch.tensor([article_to_idx[(i['source'], i['date'], i['title'])]
                    for i in each]))
        # create tensors
        self.source_idx = torch.nn.utils.rnn.pad_sequence(self.source_idx,
            batch_first=True, padding_value=-100).type(torch.int64)
        self.ideo_idx = torch.nn.utils.rnn.pad_sequence(self.ideo_idx,
            batch_first=True, padding_value=-100).type(torch.int64)
        if not loaded:
            self.tuples = torch.nn.utils.rnn.pad_sequence(self.tuples,
                batch_first=True, padding_value=-100).type(torch.int64)
            self.input_ids = torch.vstack(self.input_ids)
            self.attentions = torch.vstack(self.attentions)
        # sanity check
        assert self.source_idx.shape == self.ideo_idx.shape
        assert self.source_idx.shape == self.tuples.shape
        assert self.input_ids.shape == self.attentions.shape
        assert self.tuples.max() == self.input_ids.shape[0] - 1
        # store cache
        store_cache(inputs_file, (self.input_ids, self.attentions, self.tuples), loaded)
        
    def __len__(self):
        return self.source_idx.shape[0]

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        indices = self.tuples[idx][self.tuples[idx] != -100]
        source_idx, ideo_idx = self.source_idx[idx], self.ideo_idx[idx]
        input_ids, attentions = self.input_ids[indices], self.attentions[indices]
        
        return input_ids, attentions, source_idx[source_idx != -100], ideo_idx[ideo_idx != -100]


def get_article_entity_index(entities: List[tuple], article_ids, tokenizer, tokenizer_space):
    """Get named entity indices in the article"""
    each_start, each_end = [], []
    # only inlcude some types of entities
    entities = [i for i in entities if i[1] in ENTITY_TYPES]
    # ignore entities that are after 512 tokens
    text_len = len(tokenizer.decode(article_ids, skip_special_tokens=True))
    entities = {i[0] for i in entities if i[2] <= text_len+100} # +100 is a buffer
    for entity_i in entities:
        # entity with and without prefix space
        token_ids_no_sp = tokenizer(entity_i, return_tensors='pt',
            add_special_tokens=False)['input_ids'][0]
        token_ids_space = tokenizer_space(entity_i, return_tensors='pt',
            add_special_tokens=False)['input_ids'][0]
        for token_ids in [token_ids_no_sp, token_ids_space]:
            # if an entity is long, don't mask it because it's too hard for the model
            if token_ids.shape[0] > 5:
                continue
            window = article_ids.unfold(0, token_ids.shape[0], 1)
            # find the entity index
            start = torch.where((window == token_ids).all(dim=1))[0]
            end = start + token_ids.shape[0]
            each_start.append(start)
            each_end.append(end)
    if len(each_start) == 0:
        return torch.empty(0, 2)
    else:
        each_start = torch.cat(each_start)
        each_end = torch.cat(each_end)
        return torch.hstack([each_start.unsqueeze(1),
            each_end.unsqueeze(1)])


def get_article_sentiment_index(lexicon: Dict[int, torch.tensor], article_ids: torch.tensor):
    """Get named entity indices in the article"""
    each_start, each_end = [], []
    # single token words
    start = torch.where(article_ids == lexicon[1])[1]
    end = start + 1
    each_start.append(start)
    each_end.append(end)
    # two to five token words
    for length in [2, 3, 4, 5]:
        window = article_ids.unfold(0, length, 1).unsqueeze(0)
        lex = lexicon[length].unsqueeze(1)
        start = torch.where((window == lex).all(dim=2))[1]
        end = start + length
        each_start.append(start)
        each_end.append(end)
    # concat
    each_start = torch.cat(each_start)
    each_end = torch.cat(each_end)
    return torch.hstack([each_start.unsqueeze(1),
        each_end.unsqueeze(1)])


def load_lexicon(lexicon_dir: str):
    """Load lexicon files"""
    results = set()
    # load opinion lexicon
    dirpath = os.path.join(lexicon_dir, 'opinion-lexicon-English')
    files = os.listdir(dirpath)
    files = [os.path.join(dirpath, i) for i in files]
    polarity_map = {}
    for file_i in files:
        polarity = 'positive' if 'positive' in file_i else 'negative'
        with open(file_i, 'r') as fp:
            tmp = set(fp.read().splitlines())
        results = results.union(tmp)
        for i in tmp:
            polarity_map[i] = polarity
    # load subjectivity lexicon
    subjective = []
    with open(os.path.join(lexicon_dir,
            'subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'), 'r') as fp:
        for line in fp:
            content = line.strip().split()
            content = [i.split('=') for i in content if len(i) > 2]
            subjective.append({i[0]: i[1] for i in content})
    for i in subjective:
        polarity_map[i['word1']] = i['priorpolarity']
    # only use strong subjective words
    subjective = {i['word1'] for i in subjective if i['type'] == 'strongsubj'}
    results = results.union(subjective)
       
    print(f"Total number of lexicon: {len(results)}")
    return results


class MLMCollator:
    def __init__(self, model_name: str, replace_entity: bool=False, predict_ideology: bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.replace_entity = replace_entity
        self.predict_ideology = predict_ideology
    
    def __call__(self, batch):
        """Collate function."""
        # input_ids
        input_ids = [item[0] for item in batch]
        input_ids = torch.vstack(input_ids)
        # attentions
        attentions = [item[1] for item in batch]
        attentions = torch.vstack(attentions)
        if self.predict_ideology:
            # ideo_labels
            labels = [item[2] for item in batch]
            labels = torch.tensor(labels, dtype=torch.int64)
        # special_index
        if len(batch[0]) > 3:
            special_index = []
            for i in range(len(batch[0][3])):
                tmp = [item[3][i] for item in batch]
                special_index.append(torch.nn.utils.rnn.pad_sequence(tmp,
                    batch_first=True, padding_value=-1).type(torch.int64))
        else:
            special_index = None
        mlm_inputs, mlm_labels = mlm_mask_tokens(input_ids, self.tokenizer,
            special_indices=special_index, replace_entity=self.replace_entity)
        if self.predict_ideology:
            return mlm_inputs, attentions, mlm_labels, input_ids, labels
        else:
            return mlm_inputs, attentions, mlm_labels


class TupleCollator:
    """Collate function for tuple dataset."""
    def __init__(self, contrast_loss: str, use_story_loss: bool, n_gpu: int,
            batch_size: Optional[int]=None):
        self.contrast_loss = contrast_loss
        self.use_story_loss = use_story_loss
        self.n_gpu = n_gpu
        self.batch_size = batch_size
    
    def __call__(self, batch):
        # inputs
        input_ids = [item[0] for item in batch]
        attentions = [item[1] for item in batch]
        # source_idx
        source_idx = [item[2] for item in batch]
        # ideo_idx
        ideo_idx = [item[3] for item in batch]
        if self.contrast_loss == 'triplet':
            # sample triplets
            # import pdb; pdb.set_trace()
            ideo_trip_ind, story_trip_ind = sample_triplets(source_idx, ideo_idx, self.use_story_loss)
            if len(ideo_trip_ind) > 16 * self.n_gpu: # cannot load more on gpu
                ideo_trip_ind = random.sample(ideo_trip_ind, k=16 * self.n_gpu)
            if len(story_trip_ind) > 16 * self.n_gpu:
                story_trip_ind = random.sample(story_trip_ind, k=16 * self.n_gpu)
            # get tuple inputs
            ideo_anc, ideo_pos, ideo_neg = collect_inputs(ideo_trip_ind, input_ids, attentions)
            story_anc, story_pos, story_neg = collect_inputs(story_trip_ind, input_ids, attentions)
            return ideo_anc, ideo_pos, ideo_neg, story_anc, story_pos, story_neg
        else:
            out_input_ids = torch.zeros(self.batch_size, 11, 512, dtype=torch.int64) # 11 articles, 512 tokens
            out_attentions = torch.zeros_like(out_input_ids)
            mask = torch.zeros(self.batch_size, 11, dtype=torch.int64)
            for ind, (input_i, attention_i, source_i) in enumerate(zip(input_ids,
                    attentions, source_idx)):
                out_input_ids[ind, source_i] = input_i
                out_attentions[ind, source_i] = attention_i
                mask[ind, source_i] = 1
            return out_input_ids, out_attentions, mask.bool()


def collect_inputs(ind, input_ids, attentions):
    anc = {'input_ids': [torch.empty((0, 512))], 'attention_mask': [torch.empty((0, 512))]}
    pos = {'input_ids': [torch.empty((0, 512))], 'attention_mask': [torch.empty((0, 512))]}
    neg = {'input_ids': [torch.empty((0, 512))], 'attention_mask': [torch.empty((0, 512))]}
    for i in ind:
        anc['input_ids'].append(input_ids[i[0][0]][i[0][1]].unsqueeze(0))
        anc['attention_mask'].append(attentions[i[0][0]][i[0][1]].unsqueeze(0))
        pos['input_ids'].append(input_ids[i[1][0]][i[1][1]].unsqueeze(0))
        pos['attention_mask'].append(attentions[i[1][0]][i[1][1]].unsqueeze(0))
        neg['input_ids'].append(input_ids[i[2][0]][i[2][1]].unsqueeze(0))
        neg['attention_mask'].append(attentions[i[2][0]][i[2][1]].unsqueeze(0))
    anc = {'input_ids': torch.vstack(anc['input_ids']).to(torch.int64), 'attention_mask': torch.vstack(anc['attention_mask']).to(torch.int64)}
    pos = {'input_ids': torch.vstack(pos['input_ids']).to(torch.int64), 'attention_mask': torch.vstack(pos['attention_mask']).to(torch.int64)}
    neg = {'input_ids': torch.vstack(neg['input_ids']).to(torch.int64), 'attention_mask': torch.vstack(neg['attention_mask']).to(torch.int64)}
    return anc, pos, neg


def sample_triplets(source_idx, ideo_idx, include_story):
    """Sample ideology triplets and story triplets"""
    batch_size = len(source_idx)
    ideo_trip, story_trip = [], []
    for i_batch in range(batch_size):
        tuple_len = source_idx[i_batch].shape[0]
        for i_tuple in range(tuple_len):
            ideology = ideo_idx[i_batch][i_tuple]
            source = source_idx[i_batch][i_tuple]
            pos_ind = [(i_batch, i) for i in range(tuple_len) if ideo_idx[i_batch][i] ==
                ideology and i != i_tuple]
            if len(pos_ind) > 0:
                # get story triplets
                if include_story:
                    neg_ind = [(i, torch.where(source_idx[i] == source)[0][0]) for i in
                        range(batch_size) if i != i_batch and source in source_idx[i]]
                    if len(neg_ind) > 0:
                        story_pos_ind = pos_ind
                        if len(neg_ind) > len(pos_ind):
                            story_pos_ind = random.choices(story_pos_ind, k=len(neg_ind))
                        elif len(neg_ind) < len(pos_ind):
                            neg_ind = random.choices(neg_ind, k=len(pos_ind))
                        story_trip += list(zip([(i_batch, i_tuple)]*len(neg_ind),
                            story_pos_ind, neg_ind))
                # get ideology triplets
                if ideology != 0:
                    neg_ind = [(i_batch, i) for i in range(tuple_len) if i != i_tuple
                        and ideo_idx[i_batch][i] == -1*ideology]
                    if len(neg_ind) > 0:
                        ideo_pos_ind = pos_ind
                        if len(neg_ind) > len(pos_ind):
                            ideo_pos_ind = random.choices(ideo_pos_ind, k=len(neg_ind))
                        elif len(neg_ind) < len(pos_ind):
                            neg_ind = random.choices(neg_ind, k=len(pos_ind))
                        ideo_trip += list(zip([(i_batch, i_tuple)]*len(neg_ind),
                            ideo_pos_ind, neg_ind))
    return ideo_trip, story_trip


def mlm_mask_tokens(inputs, tokenizer, mlm_probability=0.15, special_indices=None,
        entity_mask_prob=0.3, replace_entity: bool=False):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    batch_size = labels.shape[0]
    # it's possible that all articles in a batch don't have any entity
    if special_indices is not None:
        masked_special_indices = torch.zeros(labels.shape, dtype=torch.bool)
        # separately process entity and sentiment word
        for special_index in special_indices:
            assert len(special_index.shape) == 3 # B * N * 2
            num_all_entity = (special_index[:, :, 0] != -1).sum(dim=1)
            all_entity_sep = [special_index[i, :num_all_entity[i]] for i in range(batch_size)]
            # sample which entities to mask
            entity_probability = torch.full(special_index.shape[:2], entity_mask_prob)
            padded_indices = (special_index[:, :, 0] == -1)
            entity_probability.masked_fill_(padded_indices, value=0.0)
            entity_masked_indices = torch.bernoulli(entity_probability).bool()
            entity_masked_indices = [special_index[i][entity_masked_indices[i]] for i in range(batch_size)]
            # remove overlapping and consecutive ranges
            entity_masked_indices_sep = [remove_overlapping_range(i.tolist()) for i in entity_masked_indices]
            num_selected_entity = [i.shape[0] for i in entity_masked_indices_sep]
            # add selected entities to masked indices
            selected_special_mask = get_mask_from_index(entity_masked_indices_sep, batch_size)

            # 80% of the time, we mask the entity
            mask_entity_indices = torch.bernoulli(torch.full((batch_size, max(num_selected_entity)), 0.8)).bool()
            entity_to_mask = [entity_masked_indices_sep[i][mask_entity_indices[i, :num_selected_entity[i]]] for i in range(batch_size)]
            indices_masked = get_mask_from_index(entity_to_mask, batch_size)

            # 10% of the time, we replace entity with another entity in the article or random tokens
            replace_entity_indices = torch.bernoulli(torch.full((batch_size, max(num_selected_entity)), 0.5)).bool() & ~mask_entity_indices
            entity_to_replace = [entity_masked_indices_sep[i][replace_entity_indices[i, :num_selected_entity[i]]] for i in range(batch_size)]
            indices_replaced = get_mask_from_index(entity_to_replace, batch_size).bool()
            if replace_entity:
                entity_to_replace_len = [i[:, 1] - i[:, 0] for i in entity_to_replace]
                entity_to_replace_len = torch.cat(entity_to_replace_len)
                # choose random entities in the article
                random_ind = [all_entity_sep[i][torch.randperm(num_all_entity[i])[:entity_to_replace[i].shape[0]]]
                    for i in range(batch_size)]
                random_words = [inputs[i, each[0]: each[1]] for i in range(batch_size) for each in random_ind[i]]
                random_words = torch.nn.utils.rnn.pad_sequence(random_words,
                    batch_first=True, padding_value=-1)
                random_words = torch.nn.functional.pad(random_words, (0, entity_to_replace_len.max() -
                    random_words.shape[1]), value=-1)
                # replace -1 with random token
                random_tokens = torch.randint(0, len(tokenizer), random_words.shape)
                neg_one_ind = random_words == -1
                random_words[neg_one_ind] = random_tokens[neg_one_ind]
                random_words_mask = torch.arange(random_words.shape[1]).unsqueeze(0) < entity_to_replace_len.unsqueeze(1)
                # set tokens
                torch.masked_select(random_words, random_words_mask, out=inputs[indices_replaced])
            else:
                # chose random tokens
                random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
                inputs[indices_replaced] = random_words[indices_replaced]

            # set mask entities
            inputs.masked_fill_(indices_masked, value=tokenizer.mask_token_id)

            # 10% of the time, we keep the same entity

            # add to masked_special_indices
            masked_special_indices.masked_fill_(selected_special_mask, value=True)

        # randomly mask remaining tokens
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(i, already_has_special_tokens=True) for i in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        num_valid_token = labels.numel() - special_tokens_mask.sum()
        masked_entity_num = masked_special_indices.sum()
        remain_mask_prob = max(num_valid_token * mlm_probability - masked_entity_num, 0) \
            / (num_valid_token - masked_entity_num)
        remain_probability = torch.full(labels.shape, remain_mask_prob)
        remain_probability.masked_fill_(special_tokens_mask, value=0.0)
        masked_token_indices = torch.bernoulli(remain_probability).bool() & ~masked_special_indices

        # We only compute loss on masked tokens
        labels.masked_fill_(~(masked_token_indices | masked_special_indices), value=-100)
    else:
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_token_indices = torch.bernoulli(probability_matrix).bool()

        # We only compute loss on masked tokens
        labels.masked_fill_(~masked_token_indices, value=-100)

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_token_indices
    inputs.masked_fill_(indices_replaced, value=tokenizer.mask_token_id)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_token_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def get_mask_from_index(span_index, batch_size):
    """Given start and end index, return a mask for these indices"""
    values = [torch.tensor([1, -1]).repeat(i.shape[0]) for i in span_index]
    # pad with 0 because the first token can't be entity
    entity_masked_indices = torch.nn.utils.rnn.pad_sequence(span_index,
        batch_first=True).type(torch.int64).reshape(batch_size, -1)
    # pad with 0 so that it won't affect cumsum
    values = torch.nn.utils.rnn.pad_sequence(values,
        batch_first=True).type(torch.int64)
    assert entity_masked_indices.shape == values.shape
    tmp_indices = torch.zeros((batch_size, 512), dtype=torch.int64, device=entity_masked_indices.device)
    tmp_indices.scatter_(1, entity_masked_indices, values)
    return tmp_indices.cumsum(1) # entity span is 1


class DownstreamDataset(Dataset):
    """Dataset for downstream tasks."""

    def __init__(self, files: List[str], convert_to_qa: bool=False, model_name: str='roberta-base',
            mask_entity: bool=False, return_entity: bool=False):
        super().__init__()
        data_list = []
        for file_i in files:
            if '.csv' in file_i:
                data = pd.read_csv(file_i)
            elif '.txt' in file_i:
                data = pd.read_csv(file_i, sep='\t')
            # rename columns
            if 'semeval' in file_i:
                # SemEval
                data.Text = data.Text.apply(lambda x: x.replace('#SemST',
                    '').strip())
                data.Stance.replace({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}, inplace=True)
            elif 'basil' in file_i:
                # BASIL
                data.fillna("-", inplace=True)
                if "sentiments" in file_i or 'stance' in file_i:
                    data.Stance.replace({'neg': 0, 'pos': 1, 'neu': 2}, inplace=True)
                elif "bias" in file_i:
                    data.Stance.replace({True:0, False: 1}, inplace=True)
            elif 'directed_' in file_i:
                def mask_entity_mention(row):
                    sentence = row['Text']
                    ent1, ent2 = row['Ent1'], row['Ent2']
                    ent1_token = re.search(rf"(\[\s*{re.escape(ent1)}\s*\])", sentence).group(1)
                    ent2_token = re.search(rf"(\[\s*{re.escape(ent2)}\s*\])", sentence).group(1)
                    sentence = sentence.replace(ent1_token, '<ENT1>')
                    sentence = sentence.replace(ent2_token, '<ENT2>')
                    sentence = sentence.strip()

                    return sentence

                if mask_entity:
                    data.Text = data.apply(mask_entity_mention, axis=1)
                    # sanity check
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.add_tokens(['<ENT1>', '<ENT2>'])
                    for each in data.Text:
                        tmp = tokenizer(each, return_tensors='pt')
                        tmp = tmp['input_ids'][0]
                        assert tokenizer.vocab['<ENT1>'] in tmp
                        assert tokenizer.vocab['<ENT2>'] in tmp
                if not convert_to_qa:
                    data['Target'] = ''
            elif 'hyper_partisan' in file_i:
                data.Stance.replace({'Left': 0, 'Right': 1, 'false': 2}, inplace=True)
                data['Target'] = ''
            elif 'congress_' in file_i:
                data['Target'] = ''
            elif 'ytb_' in file_i:
                data.Stance.replace({'L': 0, 'R': 1}, inplace=True)
                data['Target'] = ''
            elif 'allsides' in file_i:
                data['Target'] = ''
            elif 'TVtranscript' in file_i:
                data.Stance.replace({'L': 0, 'R': 1}, inplace=True)
                data['Target'] = ''
            elif 'twitter' in file_i:
                data.Stance.replace({'L': 0, 'C': 2, 'R': 1}, inplace=True)
                data['Target'] = ''
            elif 'political_debate' in file_i:
                data.Stance.replace({'stance1': 1, 'stance2': 0}, inplace=True)
            elif 'vast' in file_i:
                pass
            elif any([domain_i in file_i for domain_i in ['congs', 'news', 'tw', 'ytb']]):
                data['Target'] = ''
            else:
                print("Invalid data file!!")
                exit(1)
            # remove NA text
            data = data[~data.Text.isna()]
            # append to list
            data_list.append(data)
        self.data = pd.concat(data_list, ignore_index=True)
        if convert_to_qa:
            text, question, label = create_qa_data(self.data.Text, self.data.Stance,
                self.data.Ent1, self.data.Ent2, mask_entity)
            self.data = pd.DataFrame({'Text': text, 'Target': question, 'Stance': label})
        self.data.reset_index(drop=True, inplace=True)
        self.return_entity = return_entity
        if return_entity:
            self.data = self.data[['Text', 'Target', 'Stance', 'Ent1', 'Ent2']]
        else:
            self.data = self.data[['Text', 'Target', 'Stance']]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.loc[idx]
        if self.return_entity:
            return data.Text, data.Target, data.Stance, data.Ent1, data.Ent2
        else:
            return data.Text, data.Target, data.Stance
    
    def get_class_weights(self):
        classes = list(range(len(pd.unique(self.data.Stance))))
        c_w = compute_class_weight('balanced', classes=classes, y=self.data.Stance)
        print(c_w)
        return c_w
    
    def get_statistics(self):
        # get text length, number of topics/targets
        num_tokens = []
        for each in tqdm(self.data.Text):
            num_tokens.append(len(word_tokenize(each)))
        num_tokens = np.array(num_tokens)
        print(f"Average number of tokens: {num_tokens.mean()} ({num_tokens.std()})")
        print(f"Number of data: {len(self.data)}")
        print(f"Number of unique text: {len(pd.unique(self.data.Text))}")
        print(f"Number of unique targets: {len(pd.unique(self.data.Target))}")
