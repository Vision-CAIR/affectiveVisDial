import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from PIL import Image
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils.data_utils import *
from utils.eval_utils import top_filtering
import pickle
import os
import argparse
import time
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import re
from sklearn.metrics import f1_score
from evaluate import load

def proc_ques(ques):
    words = re.sub(r"([.,'!?\"()*#:;])",'',ques.lower()).replace('-', ' ').replace('/', ' ')
    return words

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
        

    # print('{:.2f}%'.format(count/len(results_pred)*100),flush=True)
    print('blue1: {:.2f}'.format(np.mean(bleus1)),flush=True)
    print('blue2: {:.2f}'.format(np.mean(bleus2)),flush=True)
    print('blue3: {:.2f}'.format(np.mean(bleus3)),flush=True)
    print('blue4: {:.2f}'.format(np.mean(bleus4)),flush=True)
    print('bart score: {:.2f}'.format(np.mean(barts)),flush=True)
    print('bert_precision: {:.2f}'.format(np.mean(bert_precision)),flush=True)
    print('bert_recall: {:.2f}'.format(np.mean(bert_recall)),flush=True)
    print('bert_f1: {:.2f}'.format(np.mean(bert_f1)),flush=True)
    print(f1_score(emotion_true, emotion_pred, average='weighted'),flush=True)

emotions = ['excitement', 'sadness', 'anger', 'contentment', 'something else', 'disgust', 'fear', 'amusement', 'awe']

def parse_option():
    parser = argparse.ArgumentParser('NLX-GPT')

    parser.add_argument('--ckpt_path', type=str, default='finetune')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--freq', type=int, default=10)
    parser.add_argument('--visual_backbone', default=False, action="store_true")
    parser.add_argument('--load_from_epoch', type=int, default=0)
    parser.add_argument('--finetune', default=False, action="store_true")
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--answerer', default=False, action="store_true")
    parser.add_argument('--dialog', default=False, action="store_true")
    args = parser.parse_args()
    return args

def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + '/' + tokenizer_name)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + '/' + model_name).to(device)  # load model with config
    opt = torch.load(ckpt_path + '/' + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch


def load_pretrained():
    if args.visual_backbone:
        model_path = 'pretrain/nle_model_11_vis'
    else:
        model_path = 'pretrain/nle_model_11'
    tokenizer_path = 'pretrain/nle_gpt2_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)  # load model with config
    return tokenizer, model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)  # save tokenizer

    unwrapped_model.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)

    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           **kwargs}

    accelerator.save(opt, ckpt_path + filename)

def evaluate_bleu(results_pred, results_gt):
    count = 0
    bleus1 = []
    bleus2 = []
    bleus3 = []
    bleus4 = []
    for idx, pred in enumerate(results_pred):
        emotion = pred['caption'].split("because")[0].strip()
        exp = pred['caption'].split("because")[-1].strip()
        emotion_gt = results_gt[idx]['gt'].split("because")[0].strip()
        exp_gt = results_gt[idx]['gt'].split("because")[-1].strip()

        if emotion == emotion_gt:
            count+=1
        try:
            bleus1.append(sentence_bleu([exp_gt.split()], exp.split(), weights=(1,0,0,0)))
            bleus2.append(sentence_bleu([exp_gt.split()], exp.split(), weights=(0.5,0.5,0,0)))
            bleus3.append(sentence_bleu([exp_gt.split()], exp.split(), weights=(0.33,0.33,0.33,0)))
            bleus4.append(sentence_bleu([exp_gt.split()], exp.split()))
        except:
            pass
    print('{:.2f}%'.format(count/len(results_pred)*100),flush=True)
    print('blue1: {:.2f}'.format(np.mean(bleus1)),flush=True)
    print('blue2: {:.2f}'.format(np.mean(bleus2)),flush=True)
    print('blue3: {:.2f}'.format(np.mean(bleus3)),flush=True)
    print('blue4: {:.2f}'.format(np.mean(bleus4)),flush=True)


class VQAXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data
        self.ids_list = range(len(self.data))

    def __getitem__(self, i):
        dialog_id = self.ids_list[i]
        sample = self.data[dialog_id]
        image = sample['img_src']

        text_a = ''
        text_a += proc_ques(sample['caption1'])
        text_a += proc_ques(sample['caption2'])
        if args.dialog:
            for ut in sample['conversation']:
                text_a += proc_ques(ut)
        text_a += proc_ques('what is the emotion')
        if args.visual_backbone:
            if args.answerer:
                answer = sample['answerer_emotion']
                text_b = proc_ques(sample['answerer_explanation'])
            else:
                answer = sample['emotion_after']
                if sample['emotion_after'] == sample['emotion_before']:
                    text_b = proc_ques(sample['explanation_before'])
                else:
                    text_b = proc_ques(sample['explanation_after'])
        else:
            answer = sample['emotion_before']
            text_b = proc_ques(sample['corrected_explanation_before'])

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>',
                                                                                         '<answer>',
                                                                                         '<explanation>'])
        
        tokens = self.tokenizer.tokenize(text_a)
        labels = [-100] * len(tokens)  # we dont want to predict the question, set to pad to ignore in XE
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is " + answer)
        answer_len = len(answer)
        tokens_b = self.tokenizer.tokenize(" because " + text_b) + [self.tokenizer.eos_token]
        exp_len = len(tokens_b)
        tokens += answer + tokens_b
        labels += [-100] + answer[
                           1:] + tokens_b  # labels will be shifted in the model, so for now set them same as tokens
        segment_ids += [a_segment_id] * answer_len
        segment_ids += [e_segment_id] * exp_len

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]

        assert len(tokens) == len(segment_ids)
        assert len(tokens) == len(labels)

        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)

        segment_ids += ([e_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if args.visual_backbone:
            genre = image.split('/')[-2]
            image_name = image.split('/')[-1]
            root_path = '/ibex/reference/CV/WikiArt/wikiart'
            img_path = os.path.join(root_path, genre, image_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                alter_img = os.listdir(os.path.join(root_path, genre))[0]
                alter_img_path = os.path.join(root_path, genre, alter_img)
                img = Image.open(alter_img_path).convert("RGB")
            img = self.transform(img)
        else:
            img = torch.empty([3,256,256])

        return (img, dialog_id, input_ids, labels, segment_ids)

    def __len__(self):
        return len(self.ids_list)


class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.ids_list = range(len(self.data))

    def __getitem__(self, i):
        dialog_id = self.ids_list[i]
        sample = self.data[dialog_id]
        image = sample['img_src']
        text_a = ''
        text_a += proc_ques(sample['caption1'])
        text_a += proc_ques(sample['caption2'])
        if args.dialog:
            for ut in sample['conversation']:
                text_a += proc_ques(ut)
        text_a += proc_ques('what is the emotion')
        if args.visual_backbone:
            if args.answerer:
                answer = sample['answerer_emotion']
                exp = proc_ques(sample['answerer_explanation'])
            else:
                answer = sample['emotion_after']
                if sample['emotion_after'] == sample['emotion_before']:
                    exp = proc_ques(sample['corrected_explanation_before'])
                else:
                    exp = proc_ques(sample['corrected_explanation_after'])
        else:
            answer = sample['emotion_before']
            exp = proc_ques(sample['corrected_explanation_before'])
        
        emotion = answer

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<question>', '<answer>', '<explanation>'])
        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is")
        answer_len = len(answer)
        tokens += answer

        segment_ids += [a_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if args.visual_backbone:
            genre = image.split('/')[-2]
            image_name = image.split('/')[-1]
            root_path = '/ibex/reference/CV/WikiArt/wikiart'
            img_path = os.path.join(root_path, genre, image_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                alter_img = os.listdir(os.path.join(root_path, genre))[0]
                alter_img_path = os.path.join(root_path, genre, alter_img)
                img = Image.open(alter_img_path).convert("RGB")
            img = self.transform(img)
        else:
            img = torch.empty([3,256,256])

        return img, dialog_id, input_ids, segment_ids, emotion, exp

    def __len__(self):
        return len(self.ids_list)


def sample_sequences(model, tokenizer, loader):
    model.eval()
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('because')
    max_len = 30
    results_gt = []

    for i, batch in enumerate(loader):

        current_output = []
        img, img_id, input_ids, segment_ids, emotion, exp = batch

        img = img.to(device)
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)

        emotion = emotion[0] if type(emotion) else emotion
        exp = exp[0] if type(exp) else exp

        img_embeddings = image_encoder(img) if args.visual_backbone else None
        
        always_exp = True

        with torch.no_grad():

            for step in range(max_len + 1):

                if step == max_len:
                    break

                outputs = model(input_ids=input_ids,
                                past_key_values=None,
                                attention_mask=None,
                                token_type_ids=segment_ids,
                                position_ids=None,
                                encoder_hidden_states=img_embeddings,
                                encoder_attention_mask=None,
                                labels=None,
                                use_cache=False,
                                return_dict=True)

                lm_logits = outputs.logits
                logits = lm_logits[0, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)

                if prev.item() in special_tokens_ids:
                    break

                # take care of when to start the <explanation> token
                if not always_exp:

                    if prev.item() != because_token:
                        new_segment = special_tokens_ids[-2]  # answer segment
                    else:
                        new_segment = special_tokens_ids[-1]  # explanation segment
                        always_exp = True
                else:
                    new_segment = special_tokens_ids[-1]  # explanation segment

                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim=1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim=1)

        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        results_full.append({"caption": proc_ques(decoded_sequences)})
        results_gt.append({"gt": proc_ques(emotion+' because '+exp)})

    return results_full, results_gt


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

args = parse_option()
if args.eval:
    device = 'cuda'
else:
    accelerator = Accelerator()
    device = accelerator.device

finetune_pretrained = args.finetune  # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
ckpt_path = args.ckpt_path
nle_data_train_path = 'data/train_data.pickle'
nle_data_eval_path = 'data/val_data.pickle'
nle_data_test_path = 'data/test_data.pkl'
max_seq_len = 400
no_sample = True
top_k = 0
top_p = 0.9
batch_size = args.batch_size  # per GPU
num_train_epochs = 100
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1
start_epoch = 0
temperature = 1

if args.visual_backbone:
    image_encoder = ImageEncoder(device).to(device)
    change_requires_grad(image_encoder, True)

if args.load_from_epoch > 0:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, args.load_from_epoch)

else:
    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        orig_num_tokens = len(tokenizer.encoder)

        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                       'additional_special_tokens': ['<question>', '<answer>',
                                                                                     '<explanation>']})

        assert len(tokenizer) == orig_num_tokens + num_new_tokens
        config = AutoConfig.from_pretrained('distilgpt2')

        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)
        config.img_size = img_size
        config.max_seq_len = max_seq_len
        config.add_cross_attention = args.visual_backbone

        model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)

print("Model Setup Ready...")

img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if args.eval:
    test_dataset = VQAXEvalDataset(path=nle_data_test_path,
                                transform=img_transform,
                                tokenizer=tokenizer,
                                max_seq_len=max_seq_len)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True)
else:
    train_dataset = VQAXTrainDataset(path=nle_data_train_path,
                                 transform=img_transform,
                                 tokenizer=tokenizer,
                                 max_seq_len=max_seq_len)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True)

    eval_dataset = VQAXEvalDataset(path=nle_data_eval_path,
                                transform=img_transform,
                                tokenizer=tokenizer,
                                max_seq_len=max_seq_len)

    val_loader = torch.utils.data.DataLoader(eval_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True)
if not args.eval:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
    warmup_steps = 0  # 0.10 * t_total
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if args.load_from_epoch > 0:
        scheduler.load_state_dict(scheduler_dic)

if args.eval:
    results_full, results_gt = sample_sequences(model, tokenizer, test_loader)
    file_pred = args.ckpt_path + '/results/' + str(args.load_from_epoch) + '_pred_test.json'
    with open(file_pred, 'w') as w:
        json.dump(results_full, w)
    file_gt = args.ckpt_path + '/results/' + str(args.load_from_epoch) + '_gt_test.json'
    with open(file_gt, 'w') as w:
        json.dump(results_gt, w)
    evaluate_all(results_full, results_gt)

else:
    for epoch in range(args.load_from_epoch, num_train_epochs):

        model.train()
        accum_loss = 0
        end = time.time()

        for step, batch in enumerate(train_loader):

            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, _, input_ids, labels, segment_ids = batch

            img_embeddings = image_encoder(img) if args.visual_backbone else None

            outputs = model(input_ids=input_ids,
                            past_key_values=None,
                            attention_mask=None,
                            token_type_ids=segment_ids,
                            position_ids=None,
                            encoder_hidden_states=img_embeddings,
                            encoder_attention_mask=None,
                            labels=labels,
                            use_cache=False,
                            return_dict=True)

            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            accum_loss += loss.item()

            # measure elapsed time
            end = time.time()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step == len(train_loader) - 1:
                    accelerator.print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}, Time: {}".format(epoch,
                                                                                        num_train_epochs,
                                                                                        step, len(train_loader),
                                                                                        accum_loss, time.time() - end),
                                    end='          ', flush=True)
                accum_loss = 0

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if epoch % args.freq ==0:
            save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path+'/')
            results_full, results_gt = sample_sequences(unwrapped_model, tokenizer, val_loader)
            file_pred = args.ckpt_path + '/results/' + str(epoch) + '_pred.json'
            with open(file_pred, 'w') as w:
                json.dump(results_full, w)
            file_gt = args.ckpt_path + '/results/' + str(epoch) + '_gt.json'
            with open(file_gt, 'w') as w:
                json.dump(results_gt, w)
            
            evaluate_bleu(results_full, results_gt)

