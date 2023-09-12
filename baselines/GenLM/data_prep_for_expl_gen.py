#!/usr/bin/env python
# coding: utf-8
import pickle as pkl
import pandas as pd

with open('data/blip_gen_captions.pkl', 'rb') as f:
    gen_captions = pkl.load(f)

def prepare_data_for_answerer(data, typ, task):
    data_cap = []
    data_res = []
    for i, d in enumerate(data):
        sen = ""
        if('emo1_emo2' in task):
            sen += '<emotion> ' + d['emotion1'] + ' <emotion> ' + d['emotion2']
        sen +=  ' <caption> ' + d['caption1'] + ' <caption> ' + d['caption2']
        if(task.endswith('gen_cap_')):
            sen += ' <caption> ' + gen_captions[d['img_src']] + " "
        if('conv' in task):
            sen += " <conversation> "
            for s in d['conversation']:
                sen += s + " "

        data_cap.append(str(sen) + "I feel " )
        
        if(d['answerer_emotion'].lower() == 'something else' or d['answerer_emotion'] == ''):
            res_sen = " neutral " + " because "
        else:
            res_sen = d['answerer_emotion'].lower()  + " because "

        if('corrected_answerer_explanation' in d.keys()):
            if(d['corrected_answerer_explanation'] == d['corrected_answerer_explanation'] and d['corrected_answerer_explanation'] != 'NA'):
                data_res.append(res_sen + " " + str(d['corrected_answerer_explanation']))
            else:
                data_res.append(res_sen)
        else:
            if(d['answerer_explanation'] == d['answerer_explanation'] and d['answerer_explanation'] != 'NA'):
                data_res.append(res_sen + " " + str(d['answerer_explanation']))
            else:
                data_res.append(res_sen)

    df = pd.DataFrame({'caption': data_cap, 'response': data_res})
    df.to_csv('data/' + typ + task + '.csv', header=True, index=False)

def prepare_data_for_questioner(data, typ, task):
    emotion_labels = []
    for x in data:
        emotion_labels.append(x['emotion_after'].lower())

    data_cap = []
    data_res = []
    i = 0
    for d in data:
        sen = ""
        if('emo1_emo2' in task):
            sen += '<emotion> ' + d['emotion1'] + ' <emotion> ' + d['emotion2']
        sen +=  ' <caption> ' + d['caption1'] + ' <caption> ' + d['caption2']
        if('gen_cap' in task):
            sen += ' <caption> ' + gen_captions[d['img_src']] + " "
        if('conv' in task):
            sen += " <conversation> "
            for s in d['conversation']:
                sen += s + " "
        data_cap.append(str(sen) + "I feel " )
        
        if(d['emotion_after'].lower() == 'something else'):
            res_sen = " neutral " + " because "
        else:
            res_sen = d['emotion_after'].lower()  + " because "

        if('corrected_explanation_after' in d.keys()):
            if(d['corrected_explanation_after'] == d['corrected_explanation_after'] and d['corrected_explanation_after'] != 'NA'):
                data_res.append(res_sen + " " + str(d['corrected_explanation_after']))
            else:
                if(d['corrected_explanation_before'] == d['corrected_explanation_before'] and d['corrected_explanation_before'] != 'NA'):
                    data_res.append(res_sen + " " + str(d['corrected_explanation_before']))
                else:
                    data_res.append(res_sen)
        else:
            if(d['explanation_after'] == d['explanation_after'] and d['explanation_after'] != 'NA'):
                data_res.append(res_sen + " " + str(d['explanation_after']))
            else:
                if(d['explanation_before'] == d['explanation_before'] and d['explanation_before'] != 'NA'):
                    data_res.append(res_sen + " " + str(d['explanation_before']))
                else:
                    data_res.append(res_sen)
        i += 1

    df = pd.DataFrame({'caption': data_cap, 'response': data_res})
    df.to_csv('data/' + typ + task + '.csv', header=True, index=False)


for splits in ['train', 'val', 'test']:
    with open('data/' + splits + '_data.pkl', 'rb') as f:
        data = pkl.load(f)

    questioner_tasks = ['_ques_aft_expl_gen_emo_gen_emo1_emo2_cap1_cap2_conv_ft_gen_cap',
                        '_ques_aft_expl_gen_emo_gen_emo1_emo2_cap1_cap2_ft_gen_cap'
                        '_ques_aft_expl_gen_emo_gen_emo1_emo2_conv_ft_gen_cap',
                        '_ques_aft_expl_gen_emo_gen_cap1_cap2_conv_ft_gen_cap',
                        '_ques_aft_emo_cap1_cap2_conv_gen_cap',
                        '_ques_aft_emo_cap1_cap2_conv',
                        '_ques_aft_expl_gen_emo_gen_cap1_cap2_conv'
                        ]
    for task in questioner_tasks:
        prepare_data_for_questioner(data, splits, task)

    answerer_tasks = ['_ans_aft_expl_gen_emo_gen_emo1_emo2_cap1_cap2_gen_cap',
                      '_ans_aft_expl_gen_emo_gen_emo1_emo2_cap1_cap2']
    for task in answerer_tasks:
        prepare_data_for_answerer(data, splits, task)
