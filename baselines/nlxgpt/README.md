# affect-visdial

# code of adapting nlx-gpt for affective visdial

## Requirements

+ Setup

```
conda env create -f environment.yaml
```

+ GPU

Our experiments are conducted on 4 NVIDIA V100 GPUs with `accelerater`.

## Train

First download [NLX-GPT pretrained model](https://drive.google.com/drive/folders/1Bfc__0HRzYPyvRe0Ur_oSbhO8dSavT4e?usp=sharing) and put in it /pretrained

```
# questioner w/o visual
accelerate launch train.py --dialog --visual_backbone --ckpt_path=path/to/ckpt
# questioner w/ visual
accelerate launch train.py --dialog --ckpt_path=path/to/ckpt
# answerer
accelerate launch train.py --dialog --answerer --visual_backbone --ckpt_path=path/to/ckpt 
```

## Inference

```
accelerate launch eval.py --ckpt_path=path/to/ckpt --load_from_epoch epoch
```

## Acknowledgments

The baseline code adopts from https://github.com/fawazsammani/nlxgpt