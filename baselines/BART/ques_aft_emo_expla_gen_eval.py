import os
import json
import random
from tqdm import tqdm
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from datasets import load_dataset
import evaluate
from nltk.tokenize import word_tokenize
from BARTScore.bart_score import BARTScorer
device = torch.device('cuda')
print("###################################################################################################")
print("Using {} ".format(device))
print("###################################################################################################")
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

data_path = 'data/'
logs_dir = 'logs/'
output_dir = 'EmoDialog/'
subtask = 'ques_aft_expl_gen_emo_gen_emo1_emo2_cap1_cap2_conv'

if('ft_gen_cap' in subtask):
    data_path += 'ft_gen/'

if(subtask.endswith('gen_cap_1')):
    task = 'image_blip_text_' + subtask
else:
    task = 'text_only_' + subtask

modelname = 'facebook/bart-large'

if(subtask.endswith('gen_cap_1')):
    numepochs = 25
else:
    numepochs = 5

max_target_length = 50
if('conv' in subtask):
    max_source_length = 350
else:
    max_source_length = 150

print("###################################################################################################")
print("Max sentence length {} ".format(max_source_length))
print("###################################################################################################")

special_tokens = {'additional_special_tokens': ['<caption>', '<emotion>']}

if(modelname == 't5-small'): 
    savename = 't5_small'
    test_batch_size = 32
    train_batch_size = 32
elif(modelname == 't5-base'): 
    savename = 't5_base'
    test_batch_size = 8
    train_batch_size = 8
elif(modelname == 't5-large'):
    savename = 't5_large'
    test_batch_size = 32
    train_batch_size = 16
elif(modelname == 't5-11b'):
    savename = 't5_11b'
    test_batch_size = 16
    train_batch_size = 16
elif(modelname == 'facebook/bart-base'): 
    savename = 'bart_small'
    test_batch_size = 64
    train_batch_size = 32
elif(modelname == 'facebook/bart-large'):
    savename = 'bart_large'
    test_batch_size = 64
    train_batch_size = 32
elif(modelname == 'facebook/opt-1.3b'):
    savename = 'opt_1.3b'
    test_batch_size = 16
    train_batch_size = 16
elif(modelname == 'facebook/opt-350m'):
    savename = 'opt_350m'
    test_batch_size = 32
    train_batch_size = 32

save_weights = os.path.join(output_dir, 'weights', savename, task, str(numepochs), 
    str(max_source_length) + '_' + str(max_target_length))
if(os.path.isdir(os.path.join(output_dir, 'weights', savename)) == False):
    os.mkdir(os.path.join(output_dir, 'weights', savename))

if(os.path.isdir(os.path.join(output_dir, 'weights', savename, task)) == False):
    os.mkdir(os.path.join(output_dir, 'weights', savename, task))

if(os.path.isdir(os.path.join(output_dir, 'weights', savename, task, str(numepochs))) == False):
    os.mkdir(os.path.join(output_dir, 'weights', savename, task, str(numepochs)))

if(os.path.isdir(save_weights) == False):
    os.mkdir(save_weights)

print("###################################################################################################")
print("Loading weights from {} ".format(save_weights))
print("###################################################################################################")

test_file =  data_path + 'test_' + subtask + '.csv'

print("###################################################################################################")
print("Testing on {} ".format(test_file))
print("###################################################################################################")

extension = test_file.split(".")[-1]
raw_datasets = load_dataset('csv', 
    data_files={'train':test_file, 'validation': test_file, 'test': test_file},
    )

tokenizer = AutoTokenizer.from_pretrained(
    save_weights,
    cache_dir=logs_dir,
    use_fast=True,
    revision='main',
    use_auth_token=None
)

num_added_toks = tokenizer.add_special_tokens(special_tokens)

model = AutoModelForSeq2SeqLM.from_pretrained(
    save_weights,
    from_tf=False,
    cache_dir=logs_dir,
    revision='main',
    use_auth_token=None,
).to(device)

model.resize_token_embeddings(len(tokenizer))

column_names = raw_datasets["train"].column_names

text_column = column_names[0]
summary_column = column_names[1]

min_target_length = 1
ignore_pad_token_for_loss = True
padding  = "max_length"
prefix = ""

def preprocess_function_test(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

test_dataset = raw_datasets["test"]
test_dataset = test_dataset.map(
              preprocess_function_test,
              batched=True,
              num_proc=None,
              remove_columns=column_names,
              load_from_cache_file=None,
              desc="Running tokenizer on validation dataset",
          )

test_batches = len(test_dataset) // test_batch_size + 1
predictions = []
for i in tqdm(range(test_batches)):
    start = i * test_batch_size
    end = min((i + 1) * test_batch_size, len(test_dataset))
    output_preds = model.generate(
                torch.LongTensor(test_dataset["input_ids"][start : end]).to(device), 
                max_length=max_target_length, 
                num_beams=None
            )
    output_preds = tokenizer.batch_decode(
            output_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    predictions += output_preds

predictions = [pred.strip() for pred in predictions]

test_df = pd.read_csv(test_file)
references = test_df['response'].tolist()

emotion_dict = {'amusement' : 0,
 'anger' : 1,
 'awe' : 2,
 'contentment' : 3,
 'disgust' : 4,
 'excitement' : 5,
 'fear' : 6,
 'sadness' : 7,
 'neutral' : 8}

references_emo = []
references_expl = []
for i, ref in enumerate(references):
    emo = ref.split()[0]
    expl = " ".join(ref.split()[2:])
    references_emo.append(int(emotion_dict[emo.strip()]))
    references_expl.append(expl)

predictions_emo = []
predictions_expl = []
for i, ref in enumerate(predictions):
    if(predictions[i]):
        emo = ref.split()[0]
        expl = " ".join(ref.split()[2:])
        if(emo.strip() not in emotion_dict):
            predictions_emo.append(int((references_emo[i] + 1) % len(emotion_dict)))
        else:
            predictions_emo.append(int(emotion_dict[emo.strip()]))
        predictions_expl.append(expl)
    else:
        predictions_emo.append(int((references_emo[i] + 1) % len(emotion_dict)))
        predictions_expl.append('')

precision, recall, f1, _ = precision_recall_fscore_support(references_emo, predictions_emo, average='weighted')
acc = accuracy_score(references_emo, predictions_emo) * 100
f1 = f1 * 100

print("Accuracy {} and F1 {} ".format(acc, f1))

bleu = evaluate.load("bleu")
bleu_results = bleu.compute(predictions=predictions_expl, references=references_expl, tokenizer=word_tokenize)
print("BLEU scores: {} ".format(bleu_results))

meteor = evaluate.load("meteor")
meteor_results = meteor.compute(predictions=predictions_expl, references=references_expl)
print("Meteor scores: {} ".format(meteor_results))

rouge = evaluate.load("rouge")
rouge_results = rouge.compute(predictions=predictions_expl, references=references_expl)
print("ROUGE scores: {} ".format(rouge_results))

bertscore = evaluate.load("bertscore")
bertscore_results = bertscore.compute(predictions=predictions_expl, references=references_expl, lang="en")
bertscore_results = sum(bertscore_results['recall']) / len(predictions)
print("BERTScore: {} ".format(bertscore_results))

bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart_score.pth')
bartscore_results = bart_scorer.score(predictions_expl, references_expl, batch_size=4)
bartscore_results = sum(bartscore_results) / len(bartscore_results)
print("BARTScore: {} ".format(bartscore_results))

save_emo_expla = os.path.join(output_dir, 'weights', savename, task, str(numepochs), 
    str(max_source_length) + '_' + str(max_target_length), 'emo_expla.txt')
with open(save_emo_expla, 'w', encoding='utf-8') as f:
    for sen in predictions:
        f.write("{}\n".format(sen))

all_metrics = {}
all_metrics['accuracy'] = acc
all_metrics['f1-weighted'] = f1
all_metrics['bleu-1'] = bleu_results['precisions'][0]
all_metrics['bleu-2'] = bleu_results['precisions'][1]
all_metrics['bleu-3'] = bleu_results['precisions'][2]
all_metrics['bleu-4'] = bleu_results['precisions'][3]
all_metrics['avg-bleu'] = bleu_results['bleu']
all_metrics['rouge'] = rouge_results['rougeL']
all_metrics['meteor'] = meteor_results['meteor']
all_metrics['bert-score'] = bertscore_results
all_metrics['bart-score'] = bartscore_results

save_res_file = os.path.join(output_dir, 'weights', savename, task, str(numepochs), 
    str(max_source_length) + '_' + str(max_target_length), 'metrics.json')

with open(save_res_file, 'w') as f:
    json.dump(all_metrics, f)
