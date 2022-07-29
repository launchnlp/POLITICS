from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss


class POLITICS(nn.Module):
    """Roberta model pretrained on news articles and media ideology."""
    def __init__(self, model_name: str, contrast_loss: Optional[str],
            batch_size: Optional[int]=16, ideo_margin: Optional[float]=1.0,
            story_margin: Optional[float]=1.0, ideo_tau: Optional[float]=0.5,
            story_tau: Optional[float]=0.5, num_article_limit: int=48,
            predict_ideology: bool=False):
        """"""
        super().__init__()
        self.lm = AutoModelForMaskedLM.from_pretrained(model_name)
        if contrast_loss == 'triplet':
            self.ideo_loss_fn = nn.TripletMarginLoss(margin=ideo_margin)
            self.story_loss_fn = nn.TripletMarginLoss(margin=story_margin)
        else:
            # infoNCE loss
            self.ideo_mask = torch.zeros(batch_size, 11, batch_size, 11,
                dtype=torch.int64, pin_memory=True)
            self.story_mask = torch.zeros(batch_size, 11, batch_size, 11,
                dtype=torch.int64, pin_memory=True)
            # each tuple has 11 articles
            for row in range(batch_size):
                for col in range(11):
                    # set ideology mask
                    if col <= 4:
                        inds = [i for i in range(5) if i != col]
                        self.ideo_mask[row, col, row, inds] = 1
                        inds = list(range(8, 11))
                        self.ideo_mask[row, col, row, inds] = -1
                    # no need to set center for ideology
                    elif col >= 8:
                        inds = [i for i in range(8, 11) if i != col]
                        self.ideo_mask[row, col, row, inds] = 1
                        inds = list(range(5))
                        self.ideo_mask[row, col, row, inds] = -1
                    # set story mask
                    inds = [i for i in range(batch_size) if i != row]
                    if col <= 4:
                        cols = list(range(5))
                    elif col <= 7:
                        cols = list(range(5, 8))
                    else:
                        cols = list(range(8, 11))
                    for col_ii in cols:
                        self.story_mask[row, col, inds, col_ii] = -1
                    # set positive index
                    if col <= 4:
                        inds = [i for i in range(5) if i != col]
                        self.story_mask[row, col, row, inds] = 1
                    elif col <= 7:
                        inds = [i for i in range(5, 8) if i != col]
                        self.story_mask[row, col, row, inds] = 1
                    else:
                        inds = [i for i in range(8, 11) if i != col]
                        self.story_mask[row, col, row, inds] = 1
            self.ideo_mask = self.ideo_mask.reshape(batch_size*11, batch_size*11)
            self.story_mask = self.story_mask.reshape(batch_size*11, batch_size*11)
            self.ideo_tau = ideo_tau
            self.story_tau = story_tau
            self.batch_size = batch_size
            self.num_article_limit = num_article_limit
        if predict_ideology:
            self.clf_head = ClassificationHead(3)
            self.predict_loss_fn = nn.CrossEntropyLoss()
        self.contrast_loss = contrast_loss
        self.predict_ideology = predict_ideology
    
    def forward(self, norm_inputs, mlm_inputs, contra_type: Optional[str]=None):
        # forward pass without mask
        if norm_inputs is not None:
            if self.contrast_loss == 'triplet':
                anc, pos, neg = norm_inputs
                anc, pos, neg = self.get_cls_token(anc), self.get_cls_token(pos), self.get_cls_token(neg) 
                # contrastive loss on CLS token
                if contra_type == 'ideo':
                    ideology_loss = self.ideo_loss_fn(anc, pos, neg)
                elif contra_type == 'story':
                    ideology_loss = self.story_loss_fn(anc, pos, neg)
            elif self.contrast_loss == 'multi-negative':
                input_ids, attentions, mask = norm_inputs
                ideology_loss = self.get_multi_negative_loss(input_ids, attentions, mask, contra_type)
            elif self.predict_ideology:
                embed = self.get_cls_token(norm_inputs[0])
                logits = self.clf_head(embed)
                ideology_loss = self.predict_loss_fn(logits, norm_inputs[1])
        else:
            ideology_loss = None
        # MLM
        if mlm_inputs is not None:
            outputs = self.lm(**mlm_inputs)
            mlm_loss = outputs.loss
        else:
            mlm_loss = None
        
        return ideology_loss, mlm_loss
    
    def get_multi_negative_loss(self, input_ids, attentions, mask, contra_type):
        """Get multi-negative loss for CL"""
        mask = mask.flatten()
        index = torch.where(mask)[0]
        # only compute on existing articles
        if contra_type == 'ideo':
            template_mask = self.ideo_mask.to(mask.device)[index]
            tau = self.ideo_tau
        elif contra_type == 'story':
            template_mask = self.story_mask.to(mask.device)[index]
            tau = self.story_tau
        pos_ind, num_pos, neg_ind, contain_ind, needed_ind = \
            self.get_pos_neg_ind(template_mask, mask)
        if contain_ind.shape[0] == 0:
            return torch.tensor(-100, device=mask.device, dtype=torch.float32)
        # only encode some articles to save memory
        needed_ind = index[needed_ind]
        # downsample to save gpu memory
        if needed_ind.shape[0] > self.num_article_limit:
            indices = torch.randperm(needed_ind.shape[0],
                device=needed_ind.device)[:needed_ind.shape[0] - self.num_article_limit]
            mask[needed_ind[indices]] = False
            index = torch.where(mask)[0]
            if contra_type == 'ideo':
                template_mask = self.ideo_mask.to(mask.device)[index]
            elif contra_type == 'story':
                template_mask = self.story_mask.to(mask.device)[index]
            pos_ind, num_pos, neg_ind, contain_ind, needed_ind = \
                self.get_pos_neg_ind(template_mask, mask)
            if contain_ind.shape[0] == 0:
                return torch.tensor(-100, device=mask.device, dtype=torch.float32)
            needed_ind = index[needed_ind]

        input_ids = input_ids.reshape(-1, 512)[needed_ind]
        attentions = attentions.reshape(-1, 512)[needed_ind]
        # encode
        embed_pact = self.get_cls_token({'input_ids': input_ids, 'attention_mask': attentions})
        # normalize embedding
        embed_pact = F.normalize(embed_pact)
        embed = torch.zeros(self.batch_size * 11, 768, dtype=embed_pact.dtype,
            device=embed_pact.device) # 11 articles per tuple, 768 embedding dimension
        embed[needed_ind] = embed_pact
        sim_score = (embed @ embed.T) / tau # B*11, B*11
        final_index = index[contain_ind]
        # let N = len(final_index)
        sim_score = sim_score[final_index]
        sim_score_exp = torch.exp(sim_score)
        # sum up scores for positive samples
        numerator = -torch.sum(sim_score * pos_ind)
        # sum up scores for negative samples
        neg_score = torch.sum(sim_score_exp * neg_ind, dim=1)
        neg_score = neg_score.unsqueeze(1).unsqueeze(1) # N * 1 * 1
        pos_score = [sim_score_exp[i][pos_ind[i]] for i in range(pos_ind.shape[0])]
        pos_score = torch.nn.utils.rnn.pad_sequence(pos_score, batch_first=True) # N * max number of pos
        denominator = (neg_score + pos_score.unsqueeze(2)).squeeze(2) # N * max number of pos
        tmp_index = torch.where(pos_score.flatten() > 0)[0]
        denominator = torch.log(denominator.flatten()[tmp_index]).sum()
        return (numerator + denominator) / num_pos.sum()
    
    def get_pos_neg_ind(self, template_mask, mask):
        """Given template mask and mask, get indices for positives and negatives"""
        # find anchors that have positive and negative
        pos_ind = (template_mask == 1) & mask
        num_pos = torch.sum(pos_ind, dim=1)
        neg_ind = (template_mask == -1) & mask
        num_neg = torch.sum(neg_ind, dim=1)
        has_pos = (num_pos > 0)
        has_neg = (num_neg > 0)
        # only compute on articles that have positive and negatice samples
        contain_ind = torch.where(has_pos & has_neg)[0]
        needed_ind = torch.where(has_pos | has_neg)[0]
        pos_ind = pos_ind[contain_ind]
        neg_ind = neg_ind[contain_ind]
        num_pos = num_pos[contain_ind]
        return pos_ind, num_pos, neg_ind, contain_ind, needed_ind
    
    def get_cls_token(self, inputs):
        """Get embedding for CLS token"""
        outputs = self.lm(**inputs, output_hidden_states=True)
        embed = outputs.hidden_states[-1][:, 0, :]
        return embed
    
    def save_mlm_model(self, path):
        """Save MLM model"""
        torch.save(self.lm.state_dict(), path)


class StandardClf(nn.Module):
    """Standard fine-tuning classifier."""
    def __init__(self, model_name: str, num_label: int, device: str):
        """"""
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.clf_head = ClassificationHead(num_label)
        self.device = device
    
    def forward(self, text, targets):
        text = list(text)
        inputs = self.tokenizer(text, padding=True, truncation=True,
            return_tensors="pt", max_length=512).to(self.device)
        # import pdb; pdb.set_trace()
        outputs = self.lm(**inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[-1][:, 0, :]
        outputs = self.clf_head(outputs)

        return outputs


class StandardLongClf(nn.Module):
    """Standard fine-tuning classifier for long documents."""
    def __init__(self, model_name: str, num_label: int, max_length: int,
            overlap_len: int, max_num_posts: int, device: str):
        """"""
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.clf_head = ClassificationHead(num_label)
        self.overlap_len = overlap_len
        self.max_len = max_length
        self.max_num_posts = max_num_posts
        self.device = device
    
    def forward(self, text, targets):
        text = list(text)
        all_inputs = {'input_ids': [], 'attention_mask': []}
        lengths = []
        tmp_inputs = self.tokenizer(text)
        for ind in range(len(text)):
            total_len = len(tmp_inputs['input_ids'][ind])
            input_ids = F.pad(torch.tensor(tmp_inputs['input_ids'][ind]),
                (0, self.max_len - total_len % self.max_len),
                value=self.tokenizer.pad_token_id)
            input_ids = input_ids.unfold(0, self.max_len, self.max_len - self.overlap_len)
            lengths.append(min(input_ids.shape[0], self.max_num_posts))
            all_inputs['input_ids'].append(input_ids[:self.max_num_posts])
            attention_mask = F.pad(torch.tensor(tmp_inputs['attention_mask'][ind]),
                (0, self.max_len - total_len % self.max_len), value=0)
            all_inputs['attention_mask'].append(attention_mask.unfold(0,
                self.max_len, self.max_len - self.overlap_len)[:self.max_num_posts])
        all_inputs = {key: torch.vstack(value).to(self.device) for key, value in all_inputs.items()}
        outputs = self.lm(**all_inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[-1][:, 0, :]
        outputs = torch.split(outputs, lengths)
        outputs = [torch.mean(i, dim=0) for i in outputs]
        outputs = torch.vstack(outputs)
        outputs = self.clf_head(outputs)

        return outputs


class PromptClf(nn.Module):
    """Prompt classifier."""
    def __init__(self, model_name: str, prompt: str, label_to_word: Dict[int, str],
            device: str):
        """"""
        super().__init__()
        prefix_space = True if 'gpt' in model_name else False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=prefix_space)
        self.max_length = 512 # set the max length for different models
        if 'gpt' in model_name:
            # set pad and mask token
            self.tokenizer.pad_token, self.tokenizer.mask_token = self.tokenizer.eos_token, self.tokenizer.eos_token
            self.lm = AutoModelForCausalLM.from_pretrained(model_name)
            assert "{mask}" not in prompt, "No need to input mask token for GPT models"
        else:
            self.lm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.prompt = prompt
        self.label_to_token_id = []
        for i in range(len(label_to_word)):
            assert len(self.tokenizer.tokenize(label_to_word[i])) == 1, "Verbalizer has to be single token"
            self.label_to_token_id.append(self.tokenizer(label_to_word[i], return_tensors='pt',
                add_special_tokens=False)['input_ids'][0][0])
        self.model_name = model_name
        self.device = device
    
    def forward(self, text, targets):
        prompts = [self.prompt.format(target=i, mask=self.tokenizer.mask_token) for i in targets]
        inputs = self.tokenizer(text, prompts, padding=True, truncation='only_first',
            max_length=self.max_length, return_tensors="pt").to(self.device)
        outputs = self.lm(**inputs)
        if 'gpt' in self.model_name:
            # get the last non mask token
            mask_token_index = torch.sum(inputs['attention_mask'], dim=1) - 1
        else:
            _, mask_token_index = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)
        mask_token_logits = outputs.logits[torch.arange(len(text)), mask_token_index]
        mlm_logits = mask_token_logits[:, self.label_to_token_id]

        return mlm_logits


class QAClf(nn.Module):
    """QA-based classifier."""
    def __init__(self, model_name: str, device: str='cpu'):
        """"""
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.clf_head = ClassificationHead(1)
        self.device = device
    
    def forward(self, text, questions):
        inputs = self.tokenizer(text, questions, padding=True, truncation='only_first',
            return_tensors="pt").to(self.device)
        outputs = self.lm(**inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[-1][:, 0, :]
        outputs = self.clf_head(outputs)

        return outputs.squeeze()


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_labels, hidden_size=768, classifier_dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        # assume input is <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
