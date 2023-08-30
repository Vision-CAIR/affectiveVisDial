import argparse
import os
import json
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import clip
import traceback
from typing import List
import torch.nn as nn
from evaluate import load
from sklearn.metrics import f1_score

emotions = ['excitement', 'sadness', 'anger', 'contentment', 'something else', 'disgust', 'fear', 'amusement', 'awe']

class BARTScorer:
    def __init__(self, device='cuda', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

def evaluate_all(results_pred, results_gt):
    bertscore = load("bertscore")
    count = 0
    bleus1 = []
    bleus2 = []
    bleus3 = []
    bleus4 = []
    barts = []
    bert_precision = []
    bert_recall = []
    bert_f1 = []
    emotion_true = []
    emotion_pred = []

    for idx, pred in enumerate(results_pred):
        emotion = pred['caption'].split("because")[0].strip()
        exp = pred['caption'].split("because")[-1].strip()
        emotion_gt = results_gt[idx]['gt'].split("because")[0].strip()
        exp_gt = results_gt[idx]['gt'].split("because")[-1].strip()

        with torch.no_grad():
            bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
            bart = bart_scorer.score(exp, exp_gt)
            barts.append(np.mean(bart))
            results = bertscore.compute(predictions=[pred['caption']], references=[results_gt[idx]['gt']], model_type="distilbert-base-uncased")
            bert_precision.append(results['precision'])
            bert_recall.append(results['recall'])
            bert_f1.append(results['f1'])

        
        emotion_true.append(emotions.index(emotion_gt))
        emotion_pred.append(emotions.index(emotion))

        bleus1.append(sentence_bleu([exp_gt.split()], exp.split(), weights=(1,0,0,0)))
        bleus2.append(sentence_bleu([exp_gt.split()], exp.split(), weights=(0.5,0.5,0,0)))
        bleus3.append(sentence_bleu([exp_gt.split()], exp.split(), weights=(0.33,0.33,0.33,0)))
        bleus4.append(sentence_bleu([exp_gt.split()], exp.split()))
        
    print('blue1: {:.2f}'.format(np.mean(bleus1)),flush=True)
    print('blue2: {:.2f}'.format(np.mean(bleus2)),flush=True)
    print('blue3: {:.2f}'.format(np.mean(bleus3)),flush=True)
    print('blue4: {:.2f}'.format(np.mean(bleus4)),flush=True)
    print('bart score: {:.2f}'.format(np.mean(barts)),flush=True)
    print('bert_precision: {:.2f}'.format(np.mean(bert_precision)),flush=True)
    print('bert_recall: {:.2f}'.format(np.mean(bert_recall)),flush=True)
    print('bert_f1: {:.2f}'.format(np.mean(bert_f1)),flush=True)
    print(f1_score(emotion_true, emotion_pred, average='weighted'),flush=True)

def parse_option():
    parser = argparse.ArgumentParser('NLX-GPT')

    parser.add_argument('--ckpt_path', type=str, default='finetune')
    parser.add_argument('--load_from_epoch', type=int, default=0)
    args = parser.parse_args()
    return args

device = 'cuda'
args = parse_option()
path = args.ckpt_path+'/results/'
if not os.path.exists(path):
    os.mkdir(path)
f_pred = path + str(args.load_from_epoch) + '_pred.json'
with open(f_pred, 'r') as w:
    results_full = json.load(w)
f_gt = path + str(args.load_from_epoch) + '_gt.json'
with open(f_gt, 'r') as w:
    results_gt = json.load(w)
evaluate_all(results_full, results_gt)